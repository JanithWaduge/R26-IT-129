import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import '../constants.dart';

// CHANGED: DetectionMode enum removed. Every capture now always runs
// BOTH Model A (no filter) and Model B (with filter) from a single
// input — there's no mode to pick anymore, so there's nothing to
// switch or forget to switch before testing.

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});
  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {

  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isCameraReady = false;
  bool _isFrontCamera = false;

  bool _isCapturing  = false;
  bool _isProcessing = false;
  bool _serverOnline = false;

  final List<Uint8List> _rawFrames = [];

  // CHANGED: single result type now — always holds both models' output.
  DualResult? _result;
  String _statusText      = 'Connecting to server...';
  double _captureProgress = 0.0;
  int    _countdown       = 3;

  // CHANGED: session-level counters (reset on hot restart / app reopen).
  // These are live, honest, non-ground-truth diagnostics — NOT the
  // research false-positive rate (see constants.dart for that).
  int _sessionTests           = 0;
  int _sessionAgreements      = 0;
  int _sessionModelATriggered = 0; // Model A confidence >= threshold
  int _sessionModelBTriggered = 0; // Model B confidence >= threshold

  // ── Capture config ───────────────────────────
  static const int kCaptureFrames     = 15;
  static const int kFrameIntervalMs   = 130;
  static const int kParallelBatchSize = 2;  // Reduced for thread-safe MediaPipe

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initAll();
  }

  Future<void> _initAll() async {
    await _checkServer();
    await _initCamera();
  }

  // ════════════════════════════════════════════
  // SERVER
  // ════════════════════════════════════════════
  Future<void> _checkServer() async {
    try {
      final res = await http
          .get(Uri.parse('$kServerUrl/health'))
          .timeout(const Duration(seconds: 5));
      if (res.statusCode == 200) {
        setState(() { _serverOnline = true; _statusText = 'Server connected ✅'; });
      } else { _setServerOffline(); }
    } catch (_) { _setServerOffline(); }
  }

  void _setServerOffline() => setState(() {
    _serverOnline = false;
    _statusText   = 'Server offline ❌ — PC server start කරන්න';
  });

  // ════════════════════════════════════════════
  // CAMERA
  // ════════════════════════════════════════════
  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _statusText = 'Camera permission denied');
      return;
    }
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        setState(() => _statusText = 'No camera found');
        return;
      }
      await _setupCamera(_getCamera(_isFrontCamera));
    } catch (e) { setState(() => _statusText = 'Camera error: $e'); }
  }

  CameraDescription _getCamera(bool front) => _cameras.firstWhere(
    (c) => c.lensDirection ==
        (front ? CameraLensDirection.front : CameraLensDirection.back),
    orElse: () => _cameras.first,
  );

  Future<void> _setupCamera(CameraDescription camDesc) async {
    if (_cameraController != null) {
      await _cameraController!.dispose();
      _cameraController = null;
    }
    if (mounted) setState(() => _isCameraReady = false);

    _cameraController = CameraController(
      camDesc,
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    try {
      await _cameraController!.initialize();
      if (!mounted) return;
      setState(() {
        _isCameraReady = true;
        _statusText = _serverOnline
            ? 'Ready — Capture button press කරන්න'
            : 'Server offline — PC server start කරන්න';
      });
    } catch (e) {
      if (mounted) setState(() => _statusText = 'Camera setup error: $e');
    }
  }

  Future<void> _switchCamera() async {
    if (_cameras.length < 2 || _isCapturing) return;
    setState(() {
      _isCapturing      = false;
      _isProcessing     = false;
      _rawFrames.clear();
      _captureProgress  = 0.0;
      _result           = null;
      _isFrontCamera    = !_isFrontCamera;
    });
    await _setupCamera(_getCamera(_isFrontCamera));
  }

  // ════════════════════════════════════════════
  // CAPTURE
  // ════════════════════════════════════════════
  Future<void> _startCapture() async {
    if (_isCapturing || !_isCameraReady || !_serverOnline) return;

    setState(() {
      _isCapturing      = true;
      _rawFrames.clear();
      _captureProgress  = 0.0;
      _result           = null;
      _countdown        = 2;
      _statusText       = '🖐 Sign hold කරන්න...';
    });

    Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted || !_isCapturing) { t.cancel(); return; }
      if (_countdown > 0) {
        setState(() => _countdown--);
      } else {
        t.cancel();
      }
    });

    final stopwatch = Stopwatch()..start();
    for (int i = 0; i < kCaptureFrames; i++) {
      if (!mounted || !_isCapturing) break;
      final frameStart = stopwatch.elapsedMilliseconds;

      try {
        final XFile xfile = await _cameraController!.takePicture();
        final bytes = await xfile.readAsBytes();
        _rawFrames.add(bytes);
      } catch (_) {}

      if (mounted) {
        setState(() => _captureProgress = _rawFrames.length / kCaptureFrames);
      }

      final elapsed = stopwatch.elapsedMilliseconds - frameStart;
      final wait = kFrameIntervalMs - elapsed;
      if (wait > 0 && i < kCaptureFrames - 1) {
        await Future.delayed(Duration(milliseconds: wait));
      }
    }
    stopwatch.stop();

    if (_rawFrames.isEmpty || !mounted) {
      setState(() {
        _isCapturing = false;
        _statusText  = 'No frames captured — try again';
      });
      return;
    }

    await _processFrames();
  }

  // ════════════════════════════════════════════
  // PROCESS — Parallel batches
  // FIXED: Uses growable list properly
  // ════════════════════════════════════════════
  Future<void> _processFrames() async {
    if (!mounted) return;
    setState(() {
      _isProcessing    = true;
      _captureProgress = 0.0;
      _statusText      = 'Keypoints extract කරනවා...';
    });

    // ✅ FIXED: Pre-allocate list with placeholders, set values by index
    final List<List<double>> frameBuffer = List<List<double>>.generate(
      _rawFrames.length,
      (_) => List<double>.filled(kNumKeypoints, 0.0),
      growable: true,
    );

    int handDetectedCount = 0;
    int processed = 0;

    for (int batchStart = 0;
         batchStart < _rawFrames.length;
         batchStart += kParallelBatchSize) {
      if (!mounted) return;

      final batchEnd = (batchStart + kParallelBatchSize).clamp(0, _rawFrames.length);
      final futures = <Future<_FrameResult>>[];

      for (int i = batchStart; i < batchEnd; i++) {
        futures.add(_extractKeypoints(_rawFrames[i], i));
      }

      final results = await Future.wait(futures);
      for (final r in results) {
        if (r.index < frameBuffer.length) {
          frameBuffer[r.index] = r.keypoints;  // ← Set by index
        }
        if (r.detected) handDetectedCount++;
      }

      processed = batchEnd;
      if (mounted) {
        setState(() {
          _captureProgress = processed / _rawFrames.length;
          _statusText      = 'Processing $processed/${_rawFrames.length}...';
        });
      }
    }

    // Pad to kSequenceLength (30) if needed
    while (frameBuffer.length < kSequenceLength) {
      frameBuffer.add(List<double>.filled(kNumKeypoints, 0.0));
    }

    await _runPrediction(frameBuffer, handDetectedCount);
  }

  Future<_FrameResult> _extractKeypoints(Uint8List bytes, int index) async {
    try {
      final res = await http.post(
        Uri.parse('$kServerUrl/predict_frame'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'image': base64Encode(bytes), 'frame_id': index}),
      ).timeout(const Duration(seconds: 8));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        return _FrameResult(
          keypoints: List<double>.from(data['keypoints']),
          detected : data['hand_detected'] as bool,
          index    : index,
        );
      }
    } catch (_) {}
    return _FrameResult(
      keypoints: List<double>.filled(kNumKeypoints, 0.0),
      detected : false,
      index    : index,
    );
  }

  // ════════════════════════════════════════════
  // PREDICT
  // ════════════════════════════════════════════
  Future<void> _runPrediction(List<List<double>> frameBuffer, int handDetectedCount) async {
    if (!mounted) return;
    setState(() => _statusText = 'Sign analyze කරනවා... 🔍');

    // CHANGED: always request both models from one capture — no mode to pick.
    try {
      final res = await http.post(
        Uri.parse('$kServerUrl/predict_sequence?filter=both'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'frames': frameBuffer}),
      ).timeout(const Duration(seconds: 15));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final modelA = _parseResult(data['model_a'], handDetectedCount);
        final modelB = _parseResult(data['model_b'], handDetectedCount);
        final agree  = data['agreement'] ?? (modelA.label == modelB.label);

        setState(() {
          _result = DualResult(
            validFrames: data['valid_frames'] ?? handDetectedCount,
            modelA: modelA,
            modelB: modelB,
            agreement: agree,
          );
          // Session-level live diagnostics (not the research FPR — see
          // constants.dart for the offline-validated figures).
          _sessionTests++;
          if (agree) _sessionAgreements++;
          if (modelA.confidence >= kConfidenceThreshold) _sessionModelATriggered++;
          if (modelB.confidence >= kConfidenceThreshold) _sessionModelBTriggered++;

          _statusText = 'Sign detected! 🎉';
        });
      } else {
        setState(() => _statusText = 'Server error — try again');
      }
    } catch (e) {
      print('❌ Prediction error: $e');
      if (mounted) setState(() => _statusText = 'Connection error — try again');
    } finally {
      if (mounted) {
        setState(() {
          _isCapturing     = false;
          _isProcessing    = false;
          _captureProgress = 0.0;
          _rawFrames.clear();
        });
      }
    }
  }

  DetectionResult _parseResult(Map<String, dynamic> data, int handFrames) {
    final top3 = (data['top3'] as List? ?? []).map((e) => Top3Item(
      label     : e['label'],
      sinhala   : e['sinhala'],
      confidence: (e['confidence'] as num).toDouble(),
    )).toList();
    return DetectionResult(
      label         : data['label'] ?? 'Unknown',
      sinhala       : data['sinhala'] ?? '',
      confidence    : (data['confidence'] as num? ?? 0.0).toDouble(),
      top3          : top3,
      handFrames    : handFrames,
      totalFrames   : kCaptureFrames,
      filtered      : data['filtered'] ?? false,
      framesRemoved : (data['frames_removed'] as num?)?.toInt() ?? 0,
    );
  }

  void _reset() {
    setState(() {
      _isCapturing      = false;
      _isProcessing     = false;
      _rawFrames.clear();
      _captureProgress  = 0.0;
      _result           = null;
      _statusText       = _serverOnline
          ? 'Ready — Capture button press කරන්න'
          : 'Server offline — PC server start කරන්න';
    });
  }

  // ════════════════════════════════════════════
  // BUILD
  // Note: this screen is a full-bleed camera viewfinder, so — like every
  // camera UI (Instagram, TikTok, Google Lens) — it intentionally stays on
  // a dark canvas for contrast against the live feed, even though the rest
  // of the app is now on a white background. Accents are the new brand
  // blue / violet / green instead of the old flat palette.
  // ════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
            _buildCameraPreview(),
            _buildTopBar(context),
            _buildCameraToggle(),
            if (_isCapturing && !_isProcessing) _buildCaptureOverlay(),
            if (_isProcessing) _buildProcessingOverlay(),
            Align(alignment: Alignment.bottomCenter, child: _buildBottomPanel()),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (!_isCameraReady || _cameraController == null) {
      // Not a full-bleed camera feed yet — sits on the app's own
      // background, so it uses the light-theme text colors.
      return Container(
        color: kBackground,
        child: Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
          const CircularProgressIndicator(color: kPrimary),
          const SizedBox(height: 16),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Text(_statusText,
                style: const TextStyle(color: kInkSoft),
                textAlign: TextAlign.center),
          ),
        ])),
      );
    }
    return SizedBox.expand(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width : _cameraController!.value.previewSize!.height,
          height: _cameraController!.value.previewSize!.width,
          child : CameraPreview(_cameraController!),
        ),
      ),
    );
  }

  Widget _buildTopBar(BuildContext context) {
    return Positioned(
      top: 0, left: 0, right: 0,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter, end: Alignment.bottomCenter,
            colors: [Colors.black87, Colors.transparent],
          ),
        ),
        child: Row(children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios, color: Colors.white),
            onPressed: () => Navigator.pop(context),
          ),
          const Expanded(child: Text('SLSL Detection',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold))),
          Icon(Icons.circle, size: 10, color: _serverOnline ? kSuccess : kError),
          const SizedBox(width: 4),
          IconButton(
            icon: const Icon(Icons.flip_camera_android_rounded, color: Colors.white),
            onPressed: _isCapturing ? null : _switchCamera,
          ),
        ]),
      ),
    );
  }

  Widget _buildCameraToggle() {
    return Positioned(
      top: 68, left: 0, right: 0,
      child: Center(
        child: GestureDetector(
          onTap: _isCapturing ? null : _switchCamera,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 7),
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(30),
              border: Border.all(color: _isFrontCamera ? kSecondary.withOpacity(0.7) : kPrimary.withOpacity(0.7)),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              _camOption(Icons.camera_rear_rounded, 'Back',  !_isFrontCamera, kPrimary),
              const SizedBox(width: 8),
              Container(width: 1, height: 18, color: Colors.white24),
              const SizedBox(width: 8),
              _camOption(Icons.camera_front_rounded, 'Front', _isFrontCamera, kSecondary),
            ]),
          ),
        ),
      ),
    );
  }

  Widget _camOption(IconData icon, String label, bool active, Color color) {
    return AnimatedOpacity(
      duration: const Duration(milliseconds: 200), opacity: active ? 1.0 : 0.4,
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, color: active ? color : Colors.white54, size: 16),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(
          color: active ? color : Colors.white54, fontSize: 12,
          fontWeight: active ? FontWeight.bold : FontWeight.normal)),
      ]),
    );
  }

  Widget _buildCaptureOverlay() {
    return Positioned.fill(
      child: Container(
        decoration: BoxDecoration(border: Border.all(color: kPrimary.withOpacity(0.8), width: 3)),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Container(
            width: 160, height: 160,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: kPrimary, width: 3),
              color: kPrimary.withOpacity(0.08),
            ),
            child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
              Text('${_rawFrames.length}',
                  style: const TextStyle(color: kPrimary, fontSize: 42, fontWeight: FontWeight.w900)),
              Text('of $kCaptureFrames frames',
                  style: const TextStyle(color: Colors.white54, fontSize: 12)),
            ]),
          ),
          const SizedBox(height: 16),
          Row(mainAxisAlignment: MainAxisAlignment.center, children: const [
            Icon(Icons.back_hand_outlined, color: Colors.white70, size: 20),
            SizedBox(width: 8),
            Text('Sign hold කරගෙන ඉන්න',
                style: TextStyle(color: Colors.white, fontSize: 15)),
          ]),
          const SizedBox(height: 12),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 60),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: _captureProgress,
                backgroundColor: Colors.white12,
                valueColor: const AlwaysStoppedAnimation(kPrimary),
                minHeight: 6,
              ),
            ),
          ),
        ]),
      ),
    );
  }

  Widget _buildProcessingOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black.withOpacity(0.80),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          const CircularProgressIndicator(color: kPrimary, strokeWidth: 3),
          const SizedBox(height: 20),
          Text(_statusText,
              style: const TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.w500),
              textAlign: TextAlign.center),
          const SizedBox(height: 14),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 48),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: _captureProgress,
                backgroundColor: Colors.white12,
                valueColor: const AlwaysStoppedAnimation(kPrimary),
                minHeight: 6,
              ),
            ),
          ),
        ]),
      ),
    );
  }

  Widget _buildBottomPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 36),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter, end: Alignment.topCenter,
          colors: [Colors.black, Colors.black.withOpacity(0.85), Colors.transparent],
          stops: const [0.0, 0.55, 1.0],
        ),
      ),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        // CHANGED: mode selector removed — every capture is always dual.
        if (!_isCapturing && !_isProcessing && _result == null)
          _buildResearchBadge(),

        if (_result != null) _buildDualResultCard(_result!),

        if (_result != null) const SizedBox(height: 10),

        if (!_isProcessing && !_isCapturing)
          Text(_statusText,
              style: const TextStyle(color: Colors.white70, fontSize: 13),
              textAlign: TextAlign.center),
        const SizedBox(height: 18),

        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          if (_result != null || _isCapturing)
            Padding(
              padding: const EdgeInsets.only(right: 24),
              child: _circleBtn(
                icon: Icons.refresh_rounded,
                color: Colors.white24, iconColor: Colors.white70,
                onTap: _reset,
              ),
            ),

          Builder(builder: (context) {
            final baseColor = _isCapturing ? kError : kPrimary;
            final canPress = !(_isCapturing || _isProcessing || !_serverOnline);
            return GestureDetector(
              onTap: canPress ? _startCapture : null,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: 74, height: 74,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: (_isProcessing || !_serverOnline)
                      ? null
                      : LinearGradient(colors: [baseColor, baseColor == kPrimary ? kSecondary : baseColor.withOpacity(0.7)]),
                  color: _isProcessing
                      ? Colors.white24
                      : !_serverOnline
                          ? Colors.grey
                          : null,
                  border: Border.all(color: Colors.white, width: 3),
                  boxShadow: [BoxShadow(
                    color: (_isCapturing ? kError : kPrimary).withOpacity(0.5),
                    blurRadius: 20,
                  )],
                ),
                child: Icon(
                  _isProcessing ? Icons.hourglass_empty_rounded
                      : _isCapturing ? Icons.stop_rounded
                      : Icons.fiber_manual_record_rounded,
                  color: Colors.white, size: 32,
                ),
              ),
            );
          }),

          if (!_serverOnline && !_isCapturing)
            Padding(
              padding: const EdgeInsets.only(left: 24),
              child: _circleBtn(
                icon: Icons.refresh_rounded,
                color: kError.withOpacity(0.3), iconColor: kError,
                onTap: _checkServer,
              ),
            ),
        ]),
        const SizedBox(height: 8),

        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Text(_isFrontCamera ? '📷 Front' : '📸 Back',
              style: TextStyle(color: _isFrontCamera ? kSecondary : kPrimary, fontSize: 11)),
          const SizedBox(width: 12),
          Icon(Icons.circle, size: 8, color: _serverOnline ? kSuccess : kError),
          const SizedBox(width: 4),
          Text(_serverOnline ? 'Server online' : 'Server offline',
              style: TextStyle(color: _serverOnline ? kSuccess : kError, fontSize: 11)),
        ]),
      ]),
    );
  }

  // CHANGED: _buildModeSelector / _modeBtn removed — there's no mode
  // to pick anymore. Shown instead, before the first capture, is a
  // small research-credential badge with the offline-validated numbers.
  Widget _buildResearchBadge() {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black54, borderRadius: BorderRadius.circular(12),
        border: Border.all(color: kWarning.withOpacity(0.35)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: const [
          Icon(Icons.science_rounded, color: kWarning, size: 15),
          SizedBox(width: 6),
          Text('Validated noise filter (offline test)',
              style: TextStyle(color: kWarning, fontSize: 11, fontWeight: FontWeight.bold)),
        ]),
        const SizedBox(height: 6),
        Text(
          'False positive rate: ${kResearchFprBaseline.toStringAsFixed(0)}% → '
          '${kResearchFprFiltered.toStringAsFixed(0)}% '
          '(${kResearchFprReductionPct.toStringAsFixed(1)}% reduction, '
          'p=${kResearchPValue.toStringAsFixed(4)})',
          style: const TextStyle(color: Colors.white70, fontSize: 11, height: 1.4),
        ),
        Text(
          'Sign accuracy: ${kResearchSignAccuracy.toStringAsFixed(1)}% · $kResearchDatasetNote',
          style: const TextStyle(color: Colors.white38, fontSize: 10),
        ),
      ]),
    );
  }

  // CHANGED: single merged card — always shows Model A and Model B
  // from the SAME capture, side by side, plus a plain-language summary
  // of what happened (agree = clean sign; disagree = filter caught
  // something Model A didn't).
  Widget _buildDualResultCard(DualResult r) {
    return Container(
      width: double.infinity, padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.92), borderRadius: BorderRadius.circular(16),
        border: Border.all(color: (r.agreement ? kSuccess : kWarning).withOpacity(0.5)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.sign_language, color: kPrimary, size: 18),
          const SizedBox(width: 8),
          const Text('One capture — both models',
              style: TextStyle(color: kPrimary, fontSize: 14, fontWeight: FontWeight.bold)),
          const Spacer(),
          Text('${r.validFrames} hand frames',
              style: const TextStyle(color: Colors.white38, fontSize: 11)),
        ]),
        const SizedBox(height: 10),
        Row(children: [
          Expanded(child: _modelCard(r.modelA, 'Model A', 'No Filter', kError)),
          const SizedBox(width: 8),
          Expanded(child: _modelCard(r.modelB, 'Model B', 'With Filter', kSuccess)),
        ]),
        const SizedBox(height: 10),
        _summaryStrip(r),
        const SizedBox(height: 10),
        _sessionStatsStrip(),
      ]),
    );
  }

  Widget _modelCard(DetectionResult r, String name, String sub, Color color) {
    final handPct = r.totalFrames > 0 ? r.handFrames / r.totalFrames : 0.0;
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1), borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.5)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(name, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.bold)),
        Text(sub,  style: TextStyle(color: color.withOpacity(0.7), fontSize: 9)),
        const SizedBox(height: 6),
        Text(r.label, style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold),
            maxLines: 1, overflow: TextOverflow.ellipsis),
        Text(r.sinhala, style: const TextStyle(color: Colors.white70, fontSize: 11),
            maxLines: 1, overflow: TextOverflow.ellipsis),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: r.confidence, backgroundColor: Colors.white12,
            valueColor: AlwaysStoppedAnimation(color), minHeight: 5,
          ),
        ),
        const SizedBox(height: 4),
        Text('${(r.confidence * 100).toStringAsFixed(1)}%',
            style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.bold)),
        const SizedBox(height: 6),
        Row(children: [
          Icon(Icons.back_hand_outlined, size: 11, color: handPct > 0.5 ? kSuccess : kWarning),
          const SizedBox(width: 3),
          Text('${r.handFrames}/${r.totalFrames}',
              style: TextStyle(color: handPct > 0.5 ? kSuccess : kWarning, fontSize: 10)),
        ]),
        if (r.filtered)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text('🧹 ${r.framesRemoved} frame(s) filtered as noise',
                style: const TextStyle(color: Colors.white54, fontSize: 9)),
          ),
      ]),
    );
  }

  // Plain-language read of THIS capture — this is the live behavior
  // check you described: same sign → agree; noisy sign → Model B
  // should stay confident/correct while Model A wavers or misfires.
  Widget _summaryStrip(DualResult r) {
    final color = r.agreement ? kSuccess : kWarning;
    final removed = r.modelB.framesRemoved;
    String message;
    if (r.agreement && removed == 0) {
      message = 'Clean sign — both models agree, filter found nothing to remove.';
    } else if (r.agreement && removed > 0) {
      message = 'Both models agree on "${r.modelA.label}", but the filter still '
          'cleaned $removed noisy frame(s) before Model B classified.';
    } else {
      message = 'Models disagree: A says "${r.modelA.label}" '
          '(${(r.modelA.confidence * 100).toStringAsFixed(0)}%), '
          'B says "${r.modelB.label}" '
          '(${(r.modelB.confidence * 100).toStringAsFixed(0)}%) after removing '
          '$removed noisy frame(s) — likely unwanted movement in this capture.';
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1), borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Row(children: [
        Icon(r.agreement ? Icons.check_circle_outline : Icons.info_outline, color: color, size: 16),
        const SizedBox(width: 8),
        Expanded(child: Text(message,
            style: TextStyle(color: color, fontSize: 11, height: 1.4))),
      ]),
    );
  }

  // Live, session-only counters — explicitly NOT the research FPR
  // (that requires ground truth accidental data — see the badge above
  // and constants.dart for the offline-measured figures).
  Widget _sessionStatsStrip() {
    if (_sessionTests == 0) return const SizedBox.shrink();
    final agreePct = (_sessionAgreements / _sessionTests * 100);
    final aTrigPct = (_sessionModelATriggered / _sessionTests * 100);
    final bTrigPct = (_sessionModelBTriggered / _sessionTests * 100);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.04),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('This session ($_sessionTests test${_sessionTests == 1 ? "" : "s"})',
            style: const TextStyle(color: Colors.white38, fontSize: 9, fontWeight: FontWeight.bold)),
        const SizedBox(height: 3),
        Text(
          'Agreement: ${agreePct.toStringAsFixed(0)}% · '
          'A triggered: ${aTrigPct.toStringAsFixed(0)}% · '
          'B triggered: ${bTrigPct.toStringAsFixed(0)}%',
          style: const TextStyle(color: Colors.white54, fontSize: 10),
        ),
      ]),
    );
  }

  Widget _circleBtn({required IconData icon, required Color color, required Color iconColor, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 52, height: 52,
        decoration: BoxDecoration(shape: BoxShape.circle, color: color),
        child: Icon(icon, color: iconColor, size: 24),
      ),
    );
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) { _cameraController?.dispose(); }
    else if (state == AppLifecycleState.resumed) { _setupCamera(_getCamera(_isFrontCamera)); }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    super.dispose();
  }
}

class _FrameResult {
  final List<double> keypoints;
  final bool         detected;
  final int          index;
  _FrameResult({required this.keypoints, required this.detected, required this.index});
}

class DetectionResult {
  final String label, sinhala;
  final double confidence;
  final List<Top3Item> top3;
  final int handFrames, totalFrames;
  final bool filtered;
  final int framesRemoved; // ADDED: how many frames the filter dropped as noise
  DetectionResult({required this.label, required this.sinhala, required this.confidence,
      required this.top3, required this.handFrames, required this.totalFrames,
      required this.filtered, this.framesRemoved = 0});
}

// CHANGED: renamed from ComparisonResult — this is no longer an optional
// "mode", it's the only result type. Added `agreement` from the server.
class DualResult {
  final int validFrames;
  final DetectionResult modelA, modelB;
  final bool agreement;
  DualResult({required this.validFrames, required this.modelA, required this.modelB,
      required this.agreement});
}

class Top3Item {
  final String label, sinhala;
  final double confidence;
  Top3Item({required this.label, required this.sinhala, required this.confidence});
}