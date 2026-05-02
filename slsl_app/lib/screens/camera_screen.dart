import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import '../constants.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  // ── Camera ───────────────────────────────────
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isCameraReady = false;
  bool _isFrontCamera = false;

  // ── State ────────────────────────────────────
  bool _isCapturing  = false;
  bool _isProcessing = false;
  bool _serverOnline = false;

  // ── Raw frame buffer (Uint8List JPEGs) ───────
  final List<Uint8List> _rawFrames = [];
  Timer? _captureTimer;

  // ── Results ──────────────────────────────────
  DetectionResult? _lastResult;
  String _statusText      = 'Connecting to server...';
  double _captureProgress = 0.0;

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

  // ── Server health check ──────────────────────
  Future<void> _checkServer() async {
    try {
      final res = await http
          .get(Uri.parse('$kServerUrl/health'))
          .timeout(const Duration(seconds: 5));
      if (res.statusCode == 200) {
        setState(() {
          _serverOnline = true;
          _statusText   = 'Server connected ✅';
        });
      } else {
        _setServerOffline();
      }
    } catch (_) {
      _setServerOffline();
    }
  }

  void _setServerOffline() {
    setState(() {
      _serverOnline = false;
      _statusText   = 'Server offline ❌ — PC එකේ server start කරන්න';
    });
  }

  // ── Camera init ──────────────────────────────
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
    } catch (e) {
      setState(() => _statusText = 'Camera error: $e');
    }
  }

  CameraDescription _getCamera(bool front) {
    return _cameras.firstWhere(
      (c) => c.lensDirection ==
          (front ? CameraLensDirection.front : CameraLensDirection.back),
      orElse: () => _cameras.first,
    );
  }

  Future<void> _setupCamera(CameraDescription camDesc) async {
    if (_cameraController != null) {
      await _cameraController!.dispose();
      _cameraController = null;
    }
    setState(() => _isCameraReady = false);

    _cameraController = CameraController(
      camDesc,
      ResolutionPreset.medium,
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
      setState(() => _statusText = 'Camera setup error: $e');
    }
  }

  // ── Switch camera ────────────────────────────
  Future<void> _switchCamera() async {
    if (_cameras.length < 2 || _isCapturing) return;
    _captureTimer?.cancel();
    setState(() {
      _isCapturing     = false;
      _isProcessing    = false;
      _rawFrames.clear();
      _captureProgress = 0.0;
      _lastResult      = null;
      _isFrontCamera   = !_isFrontCamera;
    });
    await _setupCamera(_getCamera(_isFrontCamera));
  }

  // ════════════════════════════════════════════
  // STEP 1 — Capture 30 raw JPEG frames locally
  // 100ms × 30 = 3 seconds, no network calls
  // ════════════════════════════════════════════
  void _startCapture() {
    if (_isCapturing || !_isCameraReady || !_serverOnline) return;

    setState(() {
      _isCapturing     = true;
      _rawFrames.clear();
      _captureProgress = 0.0;
      _lastResult      = null;
      _statusText      = '🖐 Sign එක 3 seconds hold කරන්න...';
    });

    _captureTimer = Timer.periodic(
      const Duration(milliseconds: 100), // 30 frames in 3 seconds
      (timer) async {
        if (_rawFrames.length >= kSequenceLength) {
          timer.cancel();
          await _processFrames(); // network calls AFTER capture
          return;
        }
        await _captureRawFrame();
        if (mounted) {
          setState(() {
            _captureProgress = _rawFrames.length / kSequenceLength;
          });
        }
      },
    );
  }

  // Capture one JPEG frame and store in memory
  Future<void> _captureRawFrame() async {
    if (_cameraController == null ||
        !_cameraController!.value.isInitialized) return;
    try {
      final XFile xfile  = await _cameraController!.takePicture();
      final Uint8List bytes = await xfile.readAsBytes();
      _rawFrames.add(bytes);
    } catch (_) {
      // Skip failed frames
    }
  }

  // ════════════════════════════════════════════
  // STEP 2 — Send each frame to server for
  //           keypoint extraction (sequentially)
  // ════════════════════════════════════════════
  Future<void> _processFrames() async {
    if (!mounted) return;
    setState(() {
      _isProcessing = true;
      _statusText   = 'Extracting keypoints... ⏳';
    });

    final List<List<double>> frameBuffer = [];
    int handDetectedCount = 0;

    for (int i = 0; i < _rawFrames.length; i++) {
      if (!mounted) return;

      // Update progress UI
      setState(() {
        _statusText = 'Processing frame ${i + 1}/${_rawFrames.length}...';
        _captureProgress = i / _rawFrames.length;
      });

      try {
        final String b64 = base64Encode(_rawFrames[i]);

        final res = await http
            .post(
              Uri.parse('$kServerUrl/predict_frame'),
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode({'image': b64, 'frame_id': i}),
            )
            .timeout(const Duration(seconds: 5));

        if (res.statusCode == 200) {
          final data      = jsonDecode(res.body);
          final keypoints = List<double>.from(data['keypoints']);
          final detected  = data['hand_detected'] as bool;
          frameBuffer.add(keypoints);
          if (detected) handDetectedCount++;
        } else {
          frameBuffer.add(List.filled(kNumKeypoints, 0.0));
        }
      } catch (_) {
        frameBuffer.add(List.filled(kNumKeypoints, 0.0));
      }
    }

    // ── STEP 3: Run final prediction ──────────
    await _runPrediction(frameBuffer, handDetectedCount);
  }

  // ════════════════════════════════════════════
  // STEP 3 — Send full sequence for prediction
  // ════════════════════════════════════════════
  Future<void> _runPrediction(
    List<List<double>> frameBuffer,
    int handDetectedCount,
  ) async {
    if (!mounted) return;
    setState(() => _statusText = 'Analyzing sign... 🔍');

    try {
      final res = await http
          .post(
            Uri.parse('$kServerUrl/predict_sequence'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'frames': frameBuffer}),
          )
          .timeout(const Duration(seconds: 15));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);

        final top3 = (data['top3'] as List).map((e) {
          return Top3Item(
            label     : e['label'],
            sinhala   : e['sinhala'],
            confidence: e['confidence'].toDouble(),
          );
        }).toList();

        if (mounted) {
          setState(() {
            _lastResult = DetectionResult(
              label      : data['label'],
              sinhala    : data['sinhala'],
              confidence : data['confidence'].toDouble(),
              top3       : top3,
              handFrames : handDetectedCount,
              totalFrames: kSequenceLength,
            );
            _statusText = 'Sign detected! 🎉';
          });
        }
      } else {
        if (mounted) setState(() => _statusText = 'Server error — try again');
      }
    } catch (e) {
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

  void _reset() {
    _captureTimer?.cancel();
    setState(() {
      _isCapturing     = false;
      _isProcessing    = false;
      _rawFrames.clear();
      _captureProgress = 0.0;
      _lastResult      = null;
      _statusText      = _serverOnline
          ? 'Ready — Capture button press කරන්න'
          : 'Server offline — PC server start කරන්න';
    });
  }

  // ══════════════════════════════════════════════
  // BUILD
  // ══════════════════════════════════════════════
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
            Align(
              alignment: Alignment.bottomCenter,
              child: _buildBottomPanel(),
            ),
          ],
        ),
      ),
    );
  }

  // ── Camera preview ───────────────────────────
  Widget _buildCameraPreview() {
    if (!_isCameraReady || _cameraController == null) {
      return Container(
        color: kBackground,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const CircularProgressIndicator(color: kPrimary),
              const SizedBox(height: 16),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Text(
                  _statusText,
                  style: const TextStyle(color: Colors.white70),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        ),
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

  // ── Top bar ──────────────────────────────────
  Widget _buildTopBar(BuildContext context) {
    return Positioned(
      top: 0, left: 0, right: 0,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin : Alignment.topCenter,
            end   : Alignment.bottomCenter,
            colors: [Colors.black87, Colors.transparent],
          ),
        ),
        child: Row(
          children: [
            IconButton(
              icon     : const Icon(Icons.arrow_back_ios, color: Colors.white),
              onPressed: () => Navigator.pop(context),
            ),
            const Expanded(
              child: Text(
                'SLSL Detection',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color     : Colors.white,
                  fontSize  : 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(right: 4),
              child: Icon(
                Icons.circle,
                size : 10,
                color: _serverOnline ? kSuccess : kError,
              ),
            ),
            IconButton(
              icon     : const Icon(Icons.flip_camera_android_rounded,
                  color: Colors.white),
              onPressed: _isCapturing ? null : _switchCamera,
            ),
          ],
        ),
      ),
    );
  }

  // ── Front/Back toggle pill ───────────────────
  Widget _buildCameraToggle() {
    return Positioned(
      top: 68, left: 0, right: 0,
      child: Center(
        child: GestureDetector(
          onTap: _isCapturing ? null : _switchCamera,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 7),
            decoration: BoxDecoration(
              color       : Colors.black54,
              borderRadius: BorderRadius.circular(30),
              border      : Border.all(
                color: _isFrontCamera
                    ? kAccent.withOpacity(0.7)
                    : kPrimary.withOpacity(0.7),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _camOption(Icons.camera_rear_rounded,
                    'Back', !_isFrontCamera, kPrimary),
                const SizedBox(width: 8),
                Container(width: 1, height: 18, color: Colors.white24),
                const SizedBox(width: 8),
                _camOption(Icons.camera_front_rounded,
                    'Front', _isFrontCamera, kAccent),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _camOption(IconData icon, String label, bool active, Color color) {
    return AnimatedOpacity(
      duration: const Duration(milliseconds: 200),
      opacity : active ? 1.0 : 0.4,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: active ? color : Colors.white54, size: 16),
          const SizedBox(width: 4),
          Text(label,
              style: TextStyle(
                color     : active ? color : Colors.white54,
                fontSize  : 12,
                fontWeight: active ? FontWeight.bold : FontWeight.normal,
              )),
        ],
      ),
    );
  }

  // ── Capture overlay (during 3s recording) ───
  Widget _buildCaptureOverlay() {
    return Positioned.fill(
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: kPrimary.withOpacity(0.7), width: 3),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width : 180,
              height: 180,
              decoration: BoxDecoration(
                border      : Border.all(color: kPrimary, width: 2),
                borderRadius: BorderRadius.circular(16),
                color       : kPrimary.withOpacity(0.05),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.back_hand_outlined,
                      color: kPrimary, size: 44),
                  const SizedBox(height: 8),
                  Text(
                    '${(_captureProgress * 100).toInt()}%',
                    style: const TextStyle(
                      color     : kPrimary,
                      fontSize  : 30,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    '${_rawFrames.length}/$kSequenceLength frames',
                    style: const TextStyle(
                      color  : Colors.white70,
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              'Sign steady hold කරන්න 🖐',
              style: TextStyle(color: Colors.white, fontSize: 15),
            ),
          ],
        ),
      ),
    );
  }

  // ── Processing overlay (after capture) ──────
  Widget _buildProcessingOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black.withOpacity(0.7),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(color: kPrimary, strokeWidth: 3),
            const SizedBox(height: 20),
            Text(
              _statusText,
              style: const TextStyle(
                color    : Colors.white,
                fontSize : 16,
                fontWeight: FontWeight.w500,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 48),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: LinearProgressIndicator(
                  value          : _captureProgress,
                  backgroundColor: Colors.white24,
                  valueColor     : const AlwaysStoppedAnimation(kPrimary),
                  minHeight      : 6,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Bottom panel ─────────────────────────────
  Widget _buildBottomPanel() {
    return Container(
      width  : double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 36),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin : Alignment.bottomCenter,
          end   : Alignment.topCenter,
          colors: [
            Colors.black,
            Colors.black.withOpacity(0.85),
            Colors.transparent,
          ],
          stops: const [0.0, 0.55, 1.0],
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_lastResult != null) _buildResultCard(),
          if (_lastResult != null) const SizedBox(height: 10),

          if (!_isProcessing)
            Text(
              _statusText,
              style: const TextStyle(color: Colors.white70, fontSize: 13),
              textAlign: TextAlign.center,
            ),
          const SizedBox(height: 10),

          if (_isCapturing && !_isProcessing)
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value          : _captureProgress,
                backgroundColor: Colors.white24,
                valueColor     : const AlwaysStoppedAnimation(kPrimary),
                minHeight      : 6,
              ),
            ),
          const SizedBox(height: 18),

          // Buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_lastResult != null || _isCapturing)
                Padding(
                  padding: const EdgeInsets.only(right: 24),
                  child: _circleBtn(
                    icon     : Icons.refresh_rounded,
                    color    : Colors.white24,
                    iconColor: Colors.white70,
                    onTap    : _reset,
                  ),
                ),

              // Main capture button
              GestureDetector(
                onTap: (_isCapturing || _isProcessing || !_serverOnline)
                    ? null
                    : _startCapture,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: 74, height: 74,
                  decoration: BoxDecoration(
                    shape : BoxShape.circle,
                    color : _isProcessing
                        ? Colors.white24
                        : _isCapturing
                            ? kError.withOpacity(0.8)
                            : !_serverOnline
                                ? Colors.grey
                                : kPrimary,
                    border: Border.all(color: Colors.white, width: 3),
                    boxShadow: [
                      BoxShadow(
                        color    : (_isCapturing ? kError : kPrimary)
                            .withOpacity(0.5),
                        blurRadius: 20,
                      ),
                    ],
                  ),
                  child: Icon(
                    _isProcessing
                        ? Icons.hourglass_empty_rounded
                        : _isCapturing
                            ? Icons.stop_rounded
                            : Icons.fiber_manual_record_rounded,
                    color: Colors.white,
                    size : 32,
                  ),
                ),
              ),

              if (!_serverOnline && !_isCapturing)
                Padding(
                  padding: const EdgeInsets.only(left: 24),
                  child: _circleBtn(
                    icon     : Icons.refresh_rounded,
                    color    : kError.withOpacity(0.3),
                    iconColor: kError,
                    onTap    : _checkServer,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 8),

          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                _isFrontCamera ? '📷 Front' : '📸 Back',
                style: TextStyle(
                  color   : _isFrontCamera ? kAccent : kPrimary,
                  fontSize: 11,
                ),
              ),
              const SizedBox(width: 12),
              Icon(Icons.circle,
                  size : 8,
                  color: _serverOnline ? kSuccess : kError),
              const SizedBox(width: 4),
              Text(
                _serverOnline ? 'Server online' : 'Server offline',
                style: TextStyle(
                  color   : _serverOnline ? kSuccess : kError,
                  fontSize: 11,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // ── Result card ──────────────────────────────
  Widget _buildResultCard() {
    final r      = _lastResult!;
    final isHigh = r.confidence >= kConfidenceThreshold;
    final color  = isHigh ? kSuccess : kWarning;
    final handPct = r.totalFrames > 0
        ? r.handFrames / r.totalFrames
        : 0.0;

    return Container(
      width  : double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color       : Colors.black.withOpacity(0.88),
        borderRadius: BorderRadius.circular(16),
        border      : Border.all(color: color.withOpacity(0.6)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.sign_language, color: color, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  r.label,
                  style: TextStyle(
                    color     : color,
                    fontSize  : 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color       : color.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(20),
                  border      : Border.all(color: color.withOpacity(0.5)),
                ),
                child: Text(
                  '${(r.confidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                    color     : color,
                    fontWeight: FontWeight.bold,
                    fontSize  : 13,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Container(
            width  : double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color       : kSurface.withOpacity(0.6),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              children: [
                const Text('🇱🇰 ', style: TextStyle(fontSize: 16)),
                Expanded(
                  child: Text(
                    r.sinhala,
                    style: const TextStyle(
                      color     : Colors.white,
                      fontSize  : 22,
                      fontWeight: FontWeight.bold,
                      height    : 1.3,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value          : r.confidence,
              backgroundColor: Colors.white12,
              valueColor     : AlwaysStoppedAnimation(color),
              minHeight      : 4,
            ),
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              Icon(Icons.back_hand_outlined,
                  size : 13,
                  color: handPct > 0.5 ? kSuccess : kWarning),
              const SizedBox(width: 4),
              Text(
                'Hand detected: ${r.handFrames}/${r.totalFrames} frames',
                style: TextStyle(
                  color   : handPct > 0.5 ? kSuccess : kWarning,
                  fontSize: 11,
                ),
              ),
            ],
          ),
          if (r.top3.length > 1) ...[
            const SizedBox(height: 10),
            const Divider(color: Colors.white12, height: 1),
            const SizedBox(height: 8),
            const Text('Other possibilities:',
                style: TextStyle(color: Colors.white54, fontSize: 11)),
            const SizedBox(height: 6),
            ...r.top3.skip(1).map((t) => Padding(
                  padding: const EdgeInsets.only(bottom: 5),
                  child: Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(t.label,
                                style: const TextStyle(
                                    color: Colors.white60, fontSize: 12)),
                            Text(t.sinhala,
                                style: const TextStyle(
                                    color: Colors.white38, fontSize: 11)),
                          ],
                        ),
                      ),
                      Text(
                        '${(t.confidence * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(
                            color: Colors.white38, fontSize: 12),
                      ),
                    ],
                  ),
                )),
          ],
        ],
      ),
    );
  }

  Widget _circleBtn({
    required IconData icon,
    required Color color,
    required Color iconColor,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width : 52, height: 52,
        decoration: BoxDecoration(shape: BoxShape.circle, color: color),
        child: Icon(icon, color: iconColor, size: 24),
      ),
    );
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null ||
        !_cameraController!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _setupCamera(_getCamera(_isFrontCamera));
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _captureTimer?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }
}

// ── Data models ──────────────────────────────

class DetectionResult {
  final String         label;
  final String         sinhala;
  final double         confidence;
  final List<Top3Item> top3;
  final int            handFrames;
  final int            totalFrames;

  DetectionResult({
    required this.label,
    required this.sinhala,
    required this.confidence,
    required this.top3,
    required this.handFrames,
    required this.totalFrames,
  });
}

class Top3Item {
  final String label;
  final String sinhala;
  final double confidence;
  Top3Item({
    required this.label,
    required this.sinhala,
    required this.confidence,
  });
}