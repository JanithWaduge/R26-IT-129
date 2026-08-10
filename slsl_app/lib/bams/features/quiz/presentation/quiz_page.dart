import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
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
      ).showSnackBar(SnackBar(content: Text(error.message)));

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
      ).showSnackBar(SnackBar(content: Text(error.message)));
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
      ).showSnackBar(SnackBar(content: Text(error.message)));
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
          title: const Text('Leave quiz?'),
          content: const Text(
            'The current session will be '
            'marked as abandoned.',
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop(false);
              },
              child: const Text('Continue'),
            ),
            FilledButton(
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
        appBar: AppBar(
          title: Text(
            'Question '
            '${_session.currentQuestionIndex + 1}'
            ' of ${_session.questionCount}',
          ),
        ),
        body: SafeArea(
          child: question == null
              ? const Center(child: Text('No current question.'))
              : ListView(
                  padding: const EdgeInsets.all(20),
                  children: [
                    LinearProgressIndicator(
                      value: _session.progressPercentage / 100,
                    ),
                    const SizedBox(height: 20),
                    Row(
                      children: [
                        Chip(label: Text(question.categoryName)),
                        const SizedBox(width: 8),
                        Chip(
                          label: Text(
                            'Difficulty '
                            '${question.difficulty}',
                          ),
                        ),
                        const SizedBox(width: 8),
                        Chip(
                          label: Text(
                            question.wasDue
                                ? 'Due Review'
                                : 'Adaptive Practice',
                          ),
                        ),
                      ],
                    ),
                    Text(
                      'Selected because: '
                      '${question.selectionReason.replaceAll('_', ' ')}',
                    ),
                    const SizedBox(height: 20),
                    // Container(
                    //   height: 230,
                    //   decoration: BoxDecoration(
                    //     borderRadius: BorderRadius.circular(20),
                    //     border: Border.all(),
                    //   ),
                    //   child: Center(
                    //     child: Column(
                    //       mainAxisSize: MainAxisSize.min,
                    //       children: [
                    //         Icon(
                    //           question.mediaUri == null
                    //               ? Icons.video_library_outlined
                    //               : Icons.play_circle,
                    //           size: 72,
                    //         ),
                    //         const SizedBox(height: 12),
                    //         Text(
                    //           question.mediaUri == null
                    //               ? 'Development placeholder: '
                    //                     'validated sign media '
                    //                     'is pending.'
                    //               : 'Sign media available',
                    //           textAlign: TextAlign.center,
                    //         ),
                    //       ],
                    //     ),
                    //   ),
                    // ),
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
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(),
                        ),
                        child: const Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.video_library_outlined, size: 72),
                              SizedBox(height: 12),
                              Text(
                                'No sign video is attached '
                                'to this question.',
                                textAlign: TextAlign.center,
                              ),
                            ],
                          ),
                        ),
                      ),
                    const SizedBox(height: 20),
                    Text(
                      'Select the correct meaning',
                      style: Theme.of(context).textTheme.titleLarge,
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
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _submitting ? null : _requestHint,
                            icon: const Icon(Icons.lightbulb),
                            label: const Text('Hint'),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _submitting ? null : _skip,
                            icon: const Icon(Icons.skip_next),
                            label: const Text('Skip'),
                          ),
                        ),
                      ],
                    ),
                    if (_submitting)
                      const Padding(
                        padding: EdgeInsets.all(24),
                        child: Center(child: CircularProgressIndicator()),
                      ),
                  ],
                ),
        ),
      ),
    );
  }
}
