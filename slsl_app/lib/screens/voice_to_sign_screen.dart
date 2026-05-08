import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import '../services/asr_service.dart';
import '../widgets/sign_avatar_widget.dart';
import '../utils/sinhala_to_english.dart';

// ══════════════════════════════════════════════════════════════════
// Voice-to-Sign Screen  —  Objective 1, Stage 1
// Records audio → sends to Whisper server → shows transcription
// ══════════════════════════════════════════════════════════════════

class VoiceToSignScreen extends StatefulWidget {
  const VoiceToSignScreen({super.key});

  @override
  State<VoiceToSignScreen> createState() => _VoiceToSignScreenState();
}

class _VoiceToSignScreenState extends State<VoiceToSignScreen>
    with SingleTickerProviderStateMixin {
  // ── State ──────────────────────────────────────────────────────
  final AudioRecorder _recorder = AudioRecorder();

  _ScreenState _screenState = _ScreenState.idle;
  AsrResult?   _result;
  String?      _errorMessage;
  String?      _selectedLanguage; // null = auto, "si" = Sinhala, "ta" = Tamil
  bool         _serverOnline = false;
  bool         _showSign    = false;
  String       _signWord    = '';

  // Pulse animation for the record button
  late AnimationController _pulseController;
  late Animation<double>   _pulseAnim;

  // ── Lifecycle ──────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();

    _pulseController = AnimationController(
      vsync  : this,
      duration: const Duration(milliseconds: 900),
    )..addStatusListener((status) {
        if (status == AnimationStatus.completed) {
          _pulseController.reverse();
        } else if (status == AnimationStatus.dismissed) {
          if (_screenState == _ScreenState.recording) _pulseController.forward();
        }
      });

    _pulseAnim = Tween<double>(begin: 1.0, end: 1.18).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _checkServer();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _recorder.dispose();
    super.dispose();
  }

  // ── Server health check ────────────────────────────────────────

  Future<void> _checkServer() async {
    final online = await AsrService.checkHealth();
    if (mounted) setState(() => _serverOnline = online);
  }

  // ── Recording logic ────────────────────────────────────────────

  Future<void> _onRecordButtonTapped() async {
    if (_screenState == _ScreenState.recording) {
      await _stopAndTranscribe();
    } else if (_screenState == _ScreenState.idle ||
               _screenState == _ScreenState.done ||
               _screenState == _ScreenState.error) {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    // Request microphone permission
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      setState(() {
        _errorMessage = 'Microphone permission denied. Please enable it in Settings.';
        _screenState  = _ScreenState.error;
      });
      return;
    }

    final dir  = await getTemporaryDirectory();
    final path = '${dir.path}/slsl_voice_input.m4a';

    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.aacLc),
      path: path,
    );

    setState(() {
      _screenState  = _ScreenState.recording;
      _result       = null;
      _errorMessage = null;
    });
    _pulseController.forward();
  }

  Future<void> _stopAndTranscribe() async {
    final path = await _recorder.stop();
    _pulseController.stop();
    _pulseController.reset();

    if (path == null) {
      setState(() {
        _errorMessage = 'Recording failed — no audio captured.';
        _screenState  = _ScreenState.error;
      });
      return;
    }

    setState(() => _screenState = _ScreenState.processing);

    try {
      final result = await AsrService.transcribe(
        File(path),
        language: _selectedLanguage,
      );
      setState(() {
        _result      = result;
        _screenState = _ScreenState.done;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Server error: $e\n\nMake sure the PC server is running.';
        _screenState  = _ScreenState.error;
      });
    }
  }

  // ── Build ──────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF020818),
      body: Stack(
        children: [
          // Ambient glow — purple for Objective 1
          Positioned(
            top: -60, right: -40,
            child: _glow(220, const Color(0xFF7B2FBE), 0.08),
          ),
          Positioned(
            bottom: 80, left: -40,
            child: _glow(180, const Color(0xFF7B2FBE), 0.05),
          ),

          SafeArea(
            child: Column(
              children: [
                _buildTopBar(context),
                if (!_serverOnline) _buildServerWarning(),
                Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _buildLanguageSelector(),
                      _buildRecordSection(),
                      _buildResultSection(),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Top bar ────────────────────────────────────────────────────

  Widget _buildTopBar(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 12, 20, 0),
      child: Row(
        children: [
          IconButton(
            icon     : const Icon(Icons.arrow_back_ios_rounded,
                color: Colors.white54, size: 20),
            onPressed: () => Navigator.maybePop(context),
          ),
          const Spacer(),
          _pillBadge('Voice → Sign'),
        ],
      ),
    );
  }

  Widget _pillBadge(String label) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          border      : Border.all(
              color: const Color(0xFF7B2FBE).withOpacity(0.4)),
          borderRadius: BorderRadius.circular(20),
          color       : const Color(0xFF7B2FBE).withOpacity(0.08),
        ),
        child: Text(label,
            style: const TextStyle(
              color      : Color(0xFFCB9BFF),
              fontSize   : 11,
              fontWeight : FontWeight.w600,
              letterSpacing: 0.4,
            )),
      );

  // ── Server offline warning ─────────────────────────────────────

  Widget _buildServerWarning() => Container(
        margin : const EdgeInsets.fromLTRB(20, 12, 20, 0),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color       : const Color(0xFFFFB703).withOpacity(0.1),
          borderRadius: BorderRadius.circular(12),
          border      : Border.all(
              color: const Color(0xFFFFB703).withOpacity(0.3)),
        ),
        child: Row(
          children: [
            const Icon(Icons.warning_amber_rounded,
                color: Color(0xFFFFB703), size: 18),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'PC server offline. Start the server then tap',
                style: TextStyle(
                    color  : Colors.white.withOpacity(0.7),
                    fontSize: 12),
              ),
            ),
            GestureDetector(
              onTap: _checkServer,
              child: const Text('Retry',
                  style: TextStyle(
                      color    : Color(0xFFFFB703),
                      fontSize : 12,
                      fontWeight: FontWeight.w700)),
            ),
          ],
        ),
      );

  // ── Language selector ──────────────────────────────────────────

  Widget _buildLanguageSelector() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: Column(
        children: [
          Text('Language',
              style: TextStyle(
                  color    : Colors.white.withOpacity(0.35),
                  fontSize : 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 2)),
          const SizedBox(height: 14),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _langChip(label: 'Auto',      code: null),
              const SizedBox(width: 10),
              _langChip(label: 'සිංහල',    code: 'si'),
              const SizedBox(width: 10),
              _langChip(label: 'தமிழ்',    code: 'ta'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _langChip({required String label, required String? code}) {
    final selected = _selectedLanguage == code;
    return GestureDetector(
      onTap: () => setState(() => _selectedLanguage = code),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding : const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        decoration: BoxDecoration(
          color       : selected
              ? const Color(0xFF7B2FBE).withOpacity(0.25)
              : Colors.white.withOpacity(0.04),
          borderRadius: BorderRadius.circular(30),
          border      : Border.all(
            color: selected
                ? const Color(0xFF7B2FBE).withOpacity(0.7)
                : Colors.white.withOpacity(0.1),
            width: selected ? 1.5 : 1,
          ),
        ),
        child: Text(label,
            style: TextStyle(
              color     : selected ? const Color(0xFFCB9BFF) : Colors.white54,
              fontSize  : 14,
              fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
            )),
      ),
    );
  }

  // ── Record button section ──────────────────────────────────────

  Widget _buildRecordSection() {
    final isRecording  = _screenState == _ScreenState.recording;
    final isProcessing = _screenState == _ScreenState.processing;

    return Column(
      children: [
        // Record button
        GestureDetector(
          onTap: isProcessing ? null : _onRecordButtonTapped,
          child: ScaleTransition(
            scale: _pulseAnim,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              width : 110,
              height: 110,
              decoration: BoxDecoration(
                shape : BoxShape.circle,
                color : isRecording
                    ? const Color(0xFFEF233C).withOpacity(0.15)
                    : const Color(0xFF7B2FBE).withOpacity(0.15),
                border: Border.all(
                  color: isRecording
                      ? const Color(0xFFEF233C)
                      : const Color(0xFF7B2FBE),
                  width: 2,
                ),
                boxShadow: [
                  BoxShadow(
                    color     : isRecording
                        ? const Color(0xFFEF233C).withOpacity(0.3)
                        : const Color(0xFF7B2FBE).withOpacity(0.3),
                    blurRadius: 30,
                    spreadRadius: 4,
                  ),
                ],
              ),
              child: isProcessing
                  ? const Center(
                      child: SizedBox(
                        width : 36,
                        height: 36,
                        child : CircularProgressIndicator(
                          color      : Color(0xFF7B2FBE),
                          strokeWidth: 3,
                        ),
                      ),
                    )
                  : Icon(
                      isRecording ? Icons.stop_rounded : Icons.mic_rounded,
                      color: isRecording
                          ? const Color(0xFFEF233C)
                          : const Color(0xFFCB9BFF),
                      size: 46,
                    ),
            ),
          ),
        ),

        const SizedBox(height: 20),

        // Status label
        Text(
          _statusLabel(),
          style: TextStyle(
            color    : Colors.white.withOpacity(0.55),
            fontSize : 14,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  String _statusLabel() {
    switch (_screenState) {
      case _ScreenState.idle:       return 'Tap to start recording';
      case _ScreenState.recording:  return 'Recording…  tap to stop';
      case _ScreenState.processing: return 'Processing audio…';
      case _ScreenState.done:       return 'Done — tap to record again';
      case _ScreenState.error:      return 'Something went wrong';
    }
  }

  // ── Result / error section ─────────────────────────────────────

  Widget _buildResultSection() {
    if (_screenState == _ScreenState.error && _errorMessage != null) {
      return _buildErrorCard();
    }
    if (_screenState == _ScreenState.done && _result != null) {
      return _buildResultCard(_result!);
    }
    // Placeholder height so the layout doesn't jump
    return const SizedBox(height: 160);
  }

  Widget _buildErrorCard() => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Container(
          width  : double.infinity,
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color       : const Color(0xFFEF233C).withOpacity(0.07),
            borderRadius: BorderRadius.circular(18),
            border      : Border.all(
                color: const Color(0xFFEF233C).withOpacity(0.3)),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.error_outline_rounded,
                  color: Color(0xFFEF233C), size: 22),
              const SizedBox(width: 12),
              Expanded(
                child: Text(_errorMessage!,
                    style: TextStyle(
                        color  : Colors.white.withOpacity(0.7),
                        fontSize: 13,
                        height : 1.5)),
              ),
            ],
          ),
        ),
      );

  Widget _buildResultCard(AsrResult r) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: Container(
        width  : double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin : Alignment.topLeft,
            end   : Alignment.bottomRight,
            colors: [
              const Color(0xFF7B2FBE).withOpacity(0.12),
              const Color(0xFF03045E).withOpacity(0.8),
            ],
          ),
          borderRadius: BorderRadius.circular(20),
          border      : Border.all(
              color: const Color(0xFF7B2FBE).withOpacity(0.3)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Language badge + confidence
            Row(
              children: [
                _pillBadge(r.languageName),
                const Spacer(),
                Text('${(r.confidence * 100).toStringAsFixed(0)}% confidence',
                    style: TextStyle(
                        color  : Colors.white.withOpacity(0.4),
                        fontSize: 11)),
              ],
            ),
            const SizedBox(height: 14),

            // Transcribed text
            Text(
              r.text.isEmpty ? '(no speech detected)' : r.text,
              style: TextStyle(
                color     : r.text.isEmpty
                    ? Colors.white38
                    : Colors.white.withOpacity(0.9),
                fontSize  : 18,
                fontWeight: FontWeight.w600,
                height    : 1.5,
              ),
            ),
            const SizedBox(height: 18),

            // Sign avatar widget — shown after tapping Translate
            if (_showSign) ...[
              const SizedBox(height: 8),
              SignAvatarWidget(signWord: _signWord),
              const SizedBox(height: 8),
            ],

            // Confidence bar
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value          : r.confidence,
                minHeight      : 4,
                backgroundColor: Colors.white12,
                valueColor     : const AlwaysStoppedAnimation<Color>(
                    Color(0xFF7B2FBE)),
              ),
            ),
            const SizedBox(height: 18),

            // Placeholder for Stage 2 — will be wired up later
            SizedBox(
              width : double.infinity,
              height: 48,
              child: ElevatedButton.icon(
                // TODO Stage 2: navigate to grammar mapping + animation
                onPressed: r.text.isEmpty ? null : () {
                  setState(() {
                    _signWord = translateToSign(r.text);
                    _showSign = true;
                  });
                },
                icon : const Icon(Icons.sign_language, size: 20),
                label: const Text('Translate to Sign Language',
                    style: TextStyle(
                        fontWeight: FontWeight.w700, letterSpacing: 0.3)),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF7B2FBE),
                  foregroundColor: Colors.white,
                  elevation      : 0,
                  shape          : RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14)),
                  disabledBackgroundColor:
                      const Color(0xFF7B2FBE).withOpacity(0.3),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Helper ─────────────────────────────────────────────────────

  Widget _glow(double size, Color color, double opacity) => Container(
        width : size,
        height: size,
        decoration: BoxDecoration(
          shape    : BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color       : color.withOpacity(opacity),
              blurRadius  : size,
              spreadRadius: size * 0.4,
            ),
          ],
        ),
      );
}

// ── Screen state enum ──────────────────────────────────────────────────────
enum _ScreenState { idle, recording, processing, done, error }
