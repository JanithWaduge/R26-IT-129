import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../../../core/theme/app_colors.dart';
import '../../gamification/domain/gamification_models.dart';
import '../data/quiz_api.dart';
import '../domain/quiz_session.dart';
import 'quiz_result_page.dart';
import '../../../shared/widgets/sign_video_player.dart';

class QuizPage extends StatefulWidget {
  const QuizPage({required this.initialSession, super.key});

  final QuizSession initialSession;

  @override
  State<QuizPage> createState() => _QuizPageState();
}

class _QuizPageState extends State<QuizPage> {
  final QuizApi _api = QuizApi();

  late QuizSession _session;
  GamificationReward? _reward;

  Stopwatch _stopwatch = Stopwatch();

  bool _submitting = false;

  @override
  void initState() {
    super.initState();

    _session = widget.initialSession;
    _startQuestionTimer();
  }

  @override
  void dispose() {
    _stopwatch.stop();
    _api.dispose();
    super.dispose();
  }

  void _startQuestionTimer() {
    _stopwatch = Stopwatch()..start();
  }

  Future<void> _submitAnswer(QuizOption option) async {
    final QuizQuestion? question = _session.currentQuestion;

    if (question == null || _submitting) {
      return;
    }

    _stopwatch.stop();

    setState(() {
      _submitting = true;
    });

    try {
      final QuizActionResult result = await _api.submitReceptiveAnswer(
        sessionId: _session.id,
        questionId: question.questionId,
        selectedSignId: option.signId,
        responseTimeMs: _stopwatch.elapsedMilliseconds,
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _session = result.session;
        _reward = result.reward ?? _reward;
        _submitting = false;
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(result.feedback.message)));

      if (_session.isFinished) {
        _openResult();
        return;
      }

      _startQuestionTimer();
    } on ApiClientException catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _submitting = false;
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(error.message), backgroundColor: AppColors.error));

      _startQuestionTimer();
    }
  }

  Future<void> _requestHint() async {
    final QuizQuestion? question = _session.currentQuestion;

    if (question == null || _submitting) {
      return;
    }

    try {
      final String message = await _api.requestHint(
        sessionId: _session.id,
        questionId: question.questionId,
      );

      if (!mounted) {
        return;
      }

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(message)));
    } on ApiClientException catch (error) {
      if (!mounted) {
        return;
      }

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(error.message), backgroundColor: AppColors.error));
    }
  }

  Future<void> _skip() async {
    final QuizQuestion? question = _session.currentQuestion;

    if (question == null || _submitting) {
      return;
    }

    setState(() {
      _submitting = true;
    });

    try {
      final QuizActionResult result = await _api.skipQuestion(
        sessionId: _session.id,
        questionId: question.questionId,
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _session = result.session;
        _reward = result.reward ?? _reward;
        _submitting = false;
      });

      if (_session.isFinished) {
        _openResult();
        return;
      }

      _startQuestionTimer();
    } on ApiClientException catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _submitting = false;
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(error.message), backgroundColor: AppColors.error));
    }
  }

  void _openResult() {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (context) {
          return QuizResultPage(session: _session, reward: _reward);
        },
      ),
    );
  }

  Future<bool> _confirmExit() async {
    final bool? shouldLeave = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          title: const Text('Leave quiz?', style: TextStyle(color: AppColors.ink)),
          content: const Text(
            'The current session will be '
            'marked as abandoned.',
            style: TextStyle(color: AppColors.inkSoft),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop(false);
              },
              child: const Text('Continue'),
            ),
            FilledButton(
              style: FilledButton.styleFrom(backgroundColor: AppColors.error),
              onPressed: () {
                Navigator.of(context).pop(true);
              },
              child: const Text('Leave'),
            ),
          ],
        );
      },
    );

    if (shouldLeave == true) {
      await _api.abandon(sessionId: _session.id);

      return true;
    }

    return false;
  }

  Widget _pill(String text, Color color) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
        decoration: BoxDecoration(
          color: color.withOpacity(0.12),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(0.28)),
        ),
        child: Text(text,
            style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600)),
      );

  @override
  Widget build(BuildContext context) {
    final QuizQuestion? question = _session.currentQuestion;

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (bool didPop, Object? result) async {
        if (didPop) {
          return;
        }

        final bool leave = await _confirmExit();

        if (leave && context.mounted) {
          Navigator.of(context).pop();
        }
      },
      child: Scaffold(
        backgroundColor: AppColors.background,
        appBar: AppBar(
          title: Text(
            'Question '
            '${_session.currentQuestionIndex + 1}'
            ' of ${_session.questionCount}',
          ),
        ),
        body: SafeArea(
          child: question == null
              ? const Center(
                  child: Text('No current question.',
                      style: TextStyle(color: AppColors.inkSoft)))
              : ListView(
                  padding: const EdgeInsets.all(20),
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: LinearProgressIndicator(
                        value: _session.progressPercentage / 100,
                        backgroundColor: AppColors.hairline,
                        valueColor: const AlwaysStoppedAnimation(AppColors.primary),
                        minHeight: 8,
                      ),
                    ),
                    const SizedBox(height: 20),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        _pill(question.categoryName, AppColors.primary),
                        _pill('Difficulty ${question.difficulty}', AppColors.secondary),
                        _pill(
                          question.wasDue ? 'Due Review' : 'Adaptive Practice',
                          question.wasDue ? AppColors.warning : AppColors.success,
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      'Selected because: '
                      '${question.selectionReason.replaceAll('_', ' ')}',
                      style: const TextStyle(color: AppColors.inkSoft, fontSize: 12),
                    ),
                    const SizedBox(height: 20),
                    if (question.mediaUri != null &&
                        question.mediaUri!.isNotEmpty)
                      SignVideoPlayer(
                        key: ValueKey<String>(question.mediaUri!),
                        mediaUri: question.mediaUri!,
                        autoPlay: true,
                        loop: true,
                      )
                    else
                      Container(
                        height: 230,
                        decoration: BoxDecoration(
                          color: AppColors.surface,
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: AppColors.hairline),
                        ),
                        child: const Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.video_library_outlined,
                                  size: 64, color: AppColors.inkSoft),
                              SizedBox(height: 12),
                              Text(
                                'No sign video is attached '
                                'to this question.',
                                style: TextStyle(color: AppColors.inkSoft),
                                textAlign: TextAlign.center,
                              ),
                            ],
                          ),
                        ),
                      ),
                    const SizedBox(height: 20),
                    const Text(
                      'Select the correct meaning',
                      style: TextStyle(
                          color: AppColors.ink, fontSize: 18, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 12),
                    ...question.options.map((QuizOption option) {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 12),
                        child: FilledButton.tonal(
                          onPressed: _submitting
                              ? null
                              : () {
                                  _submitAnswer(option);
                                },
                          style: FilledButton.styleFrom(
                            backgroundColor: Colors.white,
                            foregroundColor: AppColors.ink,
                            side: const BorderSide(color: AppColors.hairline),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(14)),
                          ),
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Text(option.text),
                          ),
                        ),
                      );
                    }),
                    const SizedBox(height: 12),
                    Text(
                      'Attempt '
                      '${question.attemptNumber} • '
                      '${question.remainingAttempts} '
                      'attempts remaining',
                      style: const TextStyle(color: AppColors.inkSoft, fontSize: 12),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _submitting ? null : _requestHint,
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppColors.warning,
                              side: BorderSide(color: AppColors.warning.withOpacity(0.5)),
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12)),
                            ),
                            icon: const Icon(Icons.lightbulb),
                            label: const Text('Hint'),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _submitting ? null : _skip,
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppColors.secondary,
                              side: BorderSide(color: AppColors.secondary.withOpacity(0.5)),
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12)),
                            ),
                            icon: const Icon(Icons.skip_next),
                            label: const Text('Skip'),
                          ),
                        ),
                      ],
                    ),
                    if (_submitting)
                      const Padding(
                        padding: EdgeInsets.all(24),
                        child: Center(
                            child: CircularProgressIndicator(color: AppColors.primary)),
                      ),
                  ],
                ),
        ),
      ),
    );
  }
}