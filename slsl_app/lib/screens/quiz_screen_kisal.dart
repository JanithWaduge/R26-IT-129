// lib/screens/quiz_screen_kisal.dart
//
// Renamed from Kisal's original views/quiz_view.dart.
// Class kept as QuizViewKisal (renamed from QuizView) and imports updated
// to point at the shared project's models/services locations.

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import '../models/sign_item_kisal.dart';
import '../services/adaptive_api_service_kisal.dart';

class QuizViewKisal extends StatefulWidget {
  final String category;
  final bool includePractice;

  const QuizViewKisal({
    super.key,
    required this.category,
    this.includePractice = false,
  });

  @override
  State<QuizViewKisal> createState() => _QuizViewKisalState();
}

class _QuizViewKisalState extends State<QuizViewKisal> {
  final AdaptiveApiServiceKisal _apiService = AdaptiveApiServiceKisal();
  final TextEditingController _answerController = TextEditingController();

  List<SignItemKisal> _quizItems = [];
  int _currentIndex = 0;
  bool _isLoading = true;
  bool _isSubmitting = false;
  String _errorMessage = '';
  String _videoErrorMessage = '';

  late DateTime _questionStartTime;
  VideoPlayerController? _videoController;
  Future<void>? _initializeVideoFuture;

  @override
  void initState() {
    super.initState();
    _loadQuizData();
  }

  Future<void> _loadQuizData() async {
    try {
      final items = await _apiService.fetchNextQuiz(
        widget.category,
        includePractice: widget.includePractice,
      );
      if (!mounted) return;
      setState(() {
        _quizItems = items;
        _isLoading = false;
        _questionStartTime = DateTime.now();
      });
      if (items.isNotEmpty) {
        _loadVideoForCurrentQuestion();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = e.toString();
        _isLoading = false;
      });
    }
  }

  String _videoAssetPathFor(SignItemKisal sign) {
    // Spaces in asset filenames get URL-encoded (%20) by Android's APK
    // packaging, which breaks AssetManager.open() lookups at runtime even
    // though the file is genuinely bundled. Using underscores in the
    // filename avoids that class of bug entirely - video files on disk
    // must be renamed to match (spaces -> underscores).
    final sanitized = sign.wordEnglish.trim().replaceAll(' ', '_');
    return 'lib/video_ass/$sanitized.mp4';
  }

  void _loadVideoForCurrentQuestion() {
    if (_quizItems.isEmpty) return;

    final currentSign = _quizItems[_currentIndex];
    final controller = VideoPlayerController.asset(
      _videoAssetPathFor(currentSign),
    );
    final oldController = _videoController;
    oldController?.dispose();

    setState(() {
      _videoController = controller;
      _videoErrorMessage = '';
      _initializeVideoFuture = controller
          .initialize()
          .then((_) {
            controller
              ..setLooping(true)
              ..play();
            if (mounted) setState(() {});
          })
          .catchError((error) {
            if (!mounted) return;
            debugPrint('VIDEO LOAD FAILED for "${currentSign.wordEnglish}" '
                'at path "${_videoAssetPathFor(currentSign)}": $error');
            setState(() {
              _videoErrorMessage =
                  'No video asset found for "${currentSign.wordEnglish}"';
            });
          });
    });
  }

  Future<void> _handleAnswerSubmission() async {
    if (_answerController.text.trim().isEmpty || _isSubmitting) return;

    setState(() => _isSubmitting = true);

    final durationSeconds =
        DateTime.now().difference(_questionStartTime).inMilliseconds / 1000.0;
    final currentSign = _quizItems[_currentIndex];

    try {
      final result = await _apiService.submitAnswer(
        signId: currentSign.signId,
        userAnswer: _answerController.text.trim(),
        responseTimeSeconds: durationSeconds,
      );

      final isCorrect = result['is_correct'] ?? false;
      final qualityScore = result['calculated_quality_score'] ?? 0;

      if (!mounted) return;
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (context) => AlertDialog(
          title: Text(
            isCorrect ? 'Excellent Recall!' : 'Let\'s Practise This More',
            style: TextStyle(color: isCorrect ? Colors.green : Colors.red),
          ),
          content: Text(
            'Correct Word: ${currentSign.wordEnglish}\n'
            'Your Response Time: ${durationSeconds.toStringAsFixed(1)}s\n'
            'Memory Retention Score: $qualityScore/5',
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                _moveToNextQuestion();
              },
              child: const Text('Continue'),
            ),
          ],
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error saving telemetry: $e')));
    } finally {
      if (mounted) setState(() => _isSubmitting = false);
    }
  }

  void _moveToNextQuestion() {
    _answerController.clear();
    if (_currentIndex + 1 < _quizItems.length) {
      setState(() {
        _currentIndex++;
        _questionStartTime = DateTime.now();
      });
      _loadVideoForCurrentQuestion();
      return;
    }

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Text('Unit Completed!'),
        content: const Text(
          'Your performance history was pushed to the adaptive scheduler model.',
        ),
        actions: [
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pop(context);
            },
            child: const Text('Back to Dashboard'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    if (_errorMessage.isNotEmpty) {
      return Scaffold(body: Center(child: Text('Error: $_errorMessage')));
    }

    final currentSign = _quizItems[_currentIndex];

    return Scaffold(
      appBar: AppBar(
        title: Text(
          '${widget.includePractice ? 'PRACTICE' : 'ADAPTIVE'} - Question ${_currentIndex + 1}/${_quizItems.length}',
        ),
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildQuestionVideo(currentSign),
            const SizedBox(height: 30),
            const Text(
              'What is the meaning of this sign in English?',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _answerController,
              decoration: const InputDecoration(
                labelText: 'Enter your textual answer...',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.translate),
              ),
              textInputAction: TextInputAction.done,
              onSubmitted: (_) => _handleAnswerSubmission(),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isSubmitting ? null : _handleAnswerSubmission,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 15),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: _isSubmitting
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text(
                      'Submit Response',
                      style: TextStyle(fontSize: 16),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildQuestionVideo(SignItemKisal currentSign) {
    return Container(
      height: 300,
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey[300]!),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        alignment: Alignment.center,
        children: [
          if (_videoErrorMessage.isNotEmpty)
            _buildMissingVideoState(currentSign)
          else
            FutureBuilder<void>(
              future: _initializeVideoFuture,
              builder: (context, snapshot) {
                final controller = _videoController;

                if (snapshot.connectionState != ConnectionState.done ||
                    controller == null ||
                    !controller.value.isInitialized) {
                  return const CircularProgressIndicator(color: Colors.white);
                }

                return AspectRatio(
                  aspectRatio: controller.value.aspectRatio,
                  child: VideoPlayer(controller),
                );
              },
            ),
          Positioned(
            left: 12,
            right: 12,
            bottom: 12,
            child: Row(
              children: [
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.55),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      currentSign.wordEnglish,
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filled(
                  tooltip: 'Replay video',
                  onPressed: _videoController == null
                      ? null
                      : () {
                          _videoController!
                            ..seekTo(Duration.zero)
                            ..play();
                        },
                  icon: const Icon(Icons.replay),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMissingVideoState(SignItemKisal currentSign) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(
            Icons.video_file_outlined,
            color: Colors.white70,
            size: 52,
          ),
          const SizedBox(height: 12),
          Text(
            _videoErrorMessage,
            style: const TextStyle(color: Colors.white),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 6),
          Text(
            _videoAssetPathFor(currentSign),
            style: const TextStyle(color: Colors.white60, fontSize: 11),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _videoController?.dispose();
    _answerController.dispose();
    super.dispose();
  }
}