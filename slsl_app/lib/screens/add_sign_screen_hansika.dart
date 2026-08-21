// lib/screens/add_sign_screen_hansika.dart
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../constants.dart';
import '../services/teacher_api_service.dart';
import '../services/teacher_session.dart';
import 'send_to_authority_screen_hansika.dart'; // NEW

enum _Stage { metadataForm, liveValidation, capturing, processing, result }

class AddSignScreenHansika extends StatefulWidget {
  const AddSignScreenHansika({super.key});
  @override
  State<AddSignScreenHansika> createState() => _AddSignScreenHansikaState();
}

class _AddSignScreenHansikaState extends State<AddSignScreenHansika>
    with WidgetsBindingObserver {
  _Stage _stage = _Stage.metadataForm;

  final _englishController = TextEditingController();
  final _sinhalaController = TextEditingController();
  String _category = 'noun';

  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isCameraReady = false;

  Timer? _validationTimer;
  bool _lastValid = false;
  String _validationMessage = 'Checking camera...';

  final List<Uint8List> _rawFrames = [];
  double _captureProgress = 0.0;
  static const int kCaptureFrames = 15;
  static const int kFrameIntervalMs = 130;

  Map<String, dynamic>? _submitResult;
  String _statusText = '';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _validationTimer?.cancel();
    _cameraController?.dispose();
    _englishController.dispose();
    _sinhalaController.dispose();
    super.dispose();
  }

  // ════════════════════════════════════════════
  // CAMERA SETUP
  // ════════════════════════════════════════════
  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _validationMessage = 'Camera permission denied');
      return;
    }
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        setState(() => _validationMessage = 'No camera found');
        return;
      }
      final cam = _cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras.first,
      );
      _cameraController = CameraController(
        cam, ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await _cameraController!.initialize();
      if (!mounted) return;
      setState(() => _isCameraReady = true);
      _startLiveValidation();
    } catch (e) {
      if (mounted) setState(() => _validationMessage = 'Camera error: $e');
    }
  }

  // ════════════════════════════════════════════
  // LIVE VALIDATION — periodic frame check
  // ════════════════════════════════════════════
  void _startLiveValidation() {
    _validationTimer = Timer.periodic(const Duration(milliseconds: 900), (_) async {
      if (!mounted || _cameraController == null || !_isCameraReady) return;
      if (_stage != _Stage.liveValidation) return;
      try {
        final xfile = await _cameraController!.takePicture();
        final bytes = await xfile.readAsBytes();
        final result = await TeacherApiService.validateFrame(base64Encode(bytes));
        if (!mounted) return;
        setState(() {
          _lastValid = result['valid'] == true;
          _validationMessage = result['message'] ?? '';
        });
      } catch (_) {}
    });
  }

  // ════════════════════════════════════════════
  // CAPTURE SEQUENCE — same timing pattern as Janith's capture
  // ════════════════════════════════════════════
  Future<void> _startCapture() async {
    if (!_lastValid || _cameraController == null) return;
    _validationTimer?.cancel();
    setState(() {
      _stage = _Stage.capturing;
      _rawFrames.clear();
      _captureProgress = 0.0;
    });

    final stopwatch = Stopwatch()..start();
    for (int i = 0; i < kCaptureFrames; i++) {
      if (!mounted) break;
      final frameStart = stopwatch.elapsedMilliseconds;
      try {
        final xfile = await _cameraController!.takePicture();
        final bytes = await xfile.readAsBytes();
        _rawFrames.add(bytes);
      } catch (_) {}
      if (mounted) setState(() => _captureProgress = _rawFrames.length / kCaptureFrames);
      final elapsed = stopwatch.elapsedMilliseconds - frameStart;
      final wait = kFrameIntervalMs - elapsed;
      if (wait > 0 && i < kCaptureFrames - 1) {
        await Future.delayed(Duration(milliseconds: wait));
      }
    }
    stopwatch.stop();

    if (_rawFrames.isEmpty || !mounted) {
      setState(() {
        _stage = _Stage.liveValidation;
        _validationMessage = 'No frames captured — try again';
      });
      _startLiveValidation();
      return;
    }
    await _processAndSubmit();
  }

  // ════════════════════════════════════════════
  // EXTRACT KEYPOINTS (via Janith's /predict_frame — HTTP reuse only)
  // then submit to your teacher-dashboard backend
  // ════════════════════════════════════════════
  Future<void> _processAndSubmit() async {
    setState(() {
      _stage = _Stage.processing;
      _statusText = 'Extracting keypoints...';
    });

    final frameBuffer = <List<double>>[];
    for (int i = 0; i < _rawFrames.length; i++) {
      final kp = await TeacherApiService.extractKeypoints(base64Encode(_rawFrames[i]), i);
      frameBuffer.add(kp);
      if (mounted) setState(() => _captureProgress = (i + 1) / _rawFrames.length);
    }
    while (frameBuffer.length < kSequenceLength) {
      frameBuffer.add(List<double>.filled(kNumKeypoints, 0.0));
    }

    setState(() => _statusText = 'Submitting for review...');

    final teacherId = TeacherSession.teacherId ?? 'unknown_teacher';
    // ── CHANGED: teacherEmail now required ──
    final teacherEmail = TeacherSession.teacherEmail ?? '';
    final result = await TeacherApiService.submitSign(
      teacherId: teacherId,
      teacherEmail: teacherEmail,
      englishWord: _englishController.text.trim(),
      sinhalaWord: _sinhalaController.text.trim(),
      category: _category,
      frames: frameBuffer,
    );

    if (!mounted) return;
    setState(() {
      _submitResult = result;
      _stage = _Stage.result;
    });
  }

  void _resetToForm() {
    setState(() {
      _stage = _Stage.metadataForm;
      _englishController.clear();
      _sinhalaController.clear();
      _category = 'noun';
      _submitResult = null;
      _rawFrames.clear();
      _captureProgress = 0.0;
    });
  }

  // ════════════════════════════════════════════
  // BUILD
  // ════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    // Camera stages are full-bleed over the live feed, so they keep a dark
    // backdrop for contrast — every other stage uses the light brand theme.
    final isCameraStage = _stage == _Stage.liveValidation || _stage == _Stage.capturing;
    return Scaffold(
      backgroundColor: isCameraStage ? Colors.black : kBackground,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text('Add New Sign', style: TextStyle(color: isCameraStage ? Colors.white : kInk, fontWeight: FontWeight.w700)),
        iconTheme: IconThemeData(color: isCameraStage ? Colors.white : kInk),
      ),
      body: SafeArea(child: _buildStage()),
    );
  }

  Widget _buildStage() {
    switch (_stage) {
      case _Stage.metadataForm:
        return _buildMetadataForm();
      case _Stage.liveValidation:
      case _Stage.capturing:
        return _buildCameraStage();
      case _Stage.processing:
        return _buildProcessingStage();
      case _Stage.result:
        return _buildResultStage();
    }
  }

  Widget _buildMetadataForm() {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Sign Details',
            style: TextStyle(color: kInk, fontSize: 20, fontWeight: FontWeight.w700)),
        const SizedBox(height: 6),
        const Text('Enter the word this sign represents before recording.',
            style: TextStyle(color: kInkSoft, fontSize: 13)),
        const SizedBox(height: 24),
        _buildTextField(_englishController, 'English word', Icons.abc_rounded),
        const SizedBox(height: 16),
        _buildTextField(_sinhalaController, 'Sinhala word (සිංහල)', Icons.translate_rounded),
        const SizedBox(height: 16),
        Row(children: [
          Expanded(child: _categoryChip('noun', 'Noun')),
          const SizedBox(width: 12),
          Expanded(child: _categoryChip('verb', 'Verb')),
        ]),
        const Spacer(),
        SizedBox(
          width: double.infinity, height: 54,
          child: DecoratedBox(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(14),
              gradient: _canProceed()
                  ? const LinearGradient(colors: [kPrimary, kSecondary])
                  : null,
              color: _canProceed() ? null : const Color(0xFFEDEFF5),
            ),
            child: ElevatedButton(
              onPressed: _canProceed() ? () {
                setState(() => _stage = _Stage.liveValidation);
                _initCamera();
              } : null,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.transparent,
                shadowColor: Colors.transparent,
                disabledBackgroundColor: Colors.transparent,
                foregroundColor: Colors.white,
                disabledForegroundColor: kInkSoft,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
              child: const Text('Continue to Camera', style: TextStyle(fontWeight: FontWeight.w700)),
            ),
          ),
        ),
      ]),
    );
  }

  bool _canProceed() =>
      _englishController.text.trim().isNotEmpty && _sinhalaController.text.trim().isNotEmpty;

  Widget _buildTextField(TextEditingController c, String label, IconData icon) {
    return TextField(
      controller: c,
      onChanged: (_) => setState(() {}),
      style: const TextStyle(color: kInk),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: const TextStyle(color: kInkSoft),
        prefixIcon: Icon(icon, color: kPrimary),
        filled: true,
        fillColor: kSurface,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
      ),
    );
  }

  Widget _categoryChip(String value, String label) {
    final selected = _category == value;
    return GestureDetector(
      onTap: () => setState(() => _category = value),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14),
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: selected ? kPrimary.withOpacity(0.12) : kSurface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: selected ? kPrimary : const Color(0xFFEDEFF5)),
        ),
        child: Text(label, style: TextStyle(
            color: selected ? kPrimary : kInkSoft, fontWeight: FontWeight.w600)),
      ),
    );
  }

  Widget _buildCameraStage() {
    return Stack(children: [
      if (_isCameraReady && _cameraController != null)
        SizedBox.expand(
          child: FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(
              width: _cameraController!.value.previewSize!.height,
              height: _cameraController!.value.previewSize!.width,
              child: CameraPreview(_cameraController!),
            ),
          ),
        )
      else
        const Center(child: CircularProgressIndicator(color: kPrimary)),

      Positioned(
        top: 16, left: 16, right: 16,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: (_lastValid ? kSuccess : kError).withOpacity(0.92),
            borderRadius: BorderRadius.circular(14),
          ),
          child: Row(children: [
            Icon(_lastValid ? Icons.check_circle : Icons.error_outline, color: Colors.white, size: 20),
            const SizedBox(width: 10),
            Expanded(child: Text(
                _stage == _Stage.capturing ? 'Recording sign...' : _validationMessage,
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 13))),
          ]),
        ),
      ),

      if (_stage == _Stage.capturing)
        Positioned(
          bottom: 40, left: 24, right: 24,
          child: Column(children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: _captureProgress, minHeight: 8,
                backgroundColor: Colors.white24,
                valueColor: const AlwaysStoppedAnimation(kPrimary),
              ),
            ),
            const SizedBox(height: 10),
            const Text('Hold the sign steady...',
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
          ]),
        )
      else
        Positioned(
          bottom: 40, left: 24, right: 24,
          child: SizedBox(
            height: 58,
            child: ElevatedButton(
              onPressed: _lastValid ? _startCapture : null,
              style: ElevatedButton.styleFrom(
                backgroundColor: kSuccess, foregroundColor: Colors.white,
                disabledBackgroundColor: Colors.white24,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              ),
              child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                Icon(Icons.fiber_manual_record_rounded),
                SizedBox(width: 10),
                Text('Capture Sign', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
              ]),
            ),
          ),
        ),
    ]);
  }

  Widget _buildProcessingStage() {
    return Center(
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        CircularProgressIndicator(value: _captureProgress, color: kPrimary),
        const SizedBox(height: 20),
        Text(_statusText, style: const TextStyle(color: kInk)),
      ]),
    );
  }

  Widget _buildResultStage() {
    final result = _submitResult ?? {};
    final status = result['status'];
    final isRejected = status == 'rejected' || result['error'] != null;
    final message = result['message'] ?? result['error'] ?? 'Something went wrong.';
    // ── NEW: check if this submission completed a batch of 5 ──
    final batchReady = result['batch_ready'] == true;
    final batchCount = result['pending_batch_count'];

    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Container(
          width: 96, height: 96,
          decoration: BoxDecoration(
            color: (isRejected ? kError : kSuccess).withOpacity(0.12),
            shape: BoxShape.circle,
          ),
          child: Icon(
            isRejected ? Icons.cancel_rounded : Icons.check_circle_rounded,
            color: isRejected ? kError : kSuccess, size: 52,
          ),
        ),
        const SizedBox(height: 20),
        Text(
          isRejected ? 'Sign Not Accepted' : 'Submitted for Review!',
          style: const TextStyle(color: kInk, fontSize: 20, fontWeight: FontWeight.w700),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 10),
        Text(message.toString(),
            style: const TextStyle(color: kInkSoft, fontSize: 13),
            textAlign: TextAlign.center),

        // ── NEW: batch-ready banner ──
        if (!isRejected && batchCount != null) ...[
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: (batchReady ? kSuccess : kWarning).withOpacity(0.10),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: (batchReady ? kSuccess : kWarning).withOpacity(0.28)),
            ),
            child: Text(
              batchReady
                  ? '$batchCount signs collected — ready to send to authority!'
                  : '$batchCount / 5 signs collected so far',
              style: TextStyle(
                  color: batchReady ? kSuccess : kWarning, fontSize: 12, fontWeight: FontWeight.w600),
              textAlign: TextAlign.center,
            ),
          ),
        ],

        const SizedBox(height: 32),

        // ── NEW: show "Send to Authority" button when batch is ready ──
        if (batchReady)
          SizedBox(
            width: double.infinity, height: 52,
            child: ElevatedButton(
              onPressed: () => Navigator.push(context,
                  MaterialPageRoute(builder: (_) => const SendToAuthorityScreenHansika())),
              style: ElevatedButton.styleFrom(
                backgroundColor: kSuccess, foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
              child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                Icon(Icons.send_rounded),
                SizedBox(width: 10),
                Text('Send Batch to Authority', style: TextStyle(fontWeight: FontWeight.w700)),
              ]),
            ),
          ),
        if (batchReady) const SizedBox(height: 10),

        SizedBox(
          width: double.infinity, height: 52,
          child: DecoratedBox(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(14),
              gradient: const LinearGradient(colors: [kPrimary, kSecondary]),
            ),
            child: ElevatedButton(
              onPressed: _resetToForm,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.transparent, shadowColor: Colors.transparent,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
              child: const Text('Add Another Sign', style: TextStyle(fontWeight: FontWeight.w700)),
            ),
          ),
        ),
        const SizedBox(height: 10),
        SizedBox(
          width: double.infinity, height: 52,
          child: OutlinedButton(
            onPressed: () => Navigator.pop(context),
            style: OutlinedButton.styleFrom(
              side: const BorderSide(color: Color(0xFFEDEFF5)),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            ),
            child: const Text('Done', style: TextStyle(color: kInk)),
          ),
        ),
      ]),
    );
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    }
  }
}