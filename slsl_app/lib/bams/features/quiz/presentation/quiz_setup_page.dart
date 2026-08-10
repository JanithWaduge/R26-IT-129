import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../../vocabulary/data/vocabulary_api.dart';
import '../../vocabulary/domain/curriculum_category.dart';
import '../data/quiz_api.dart';
import '../domain/quiz_session.dart';
import 'quiz_page.dart';

class QuizSetupPage extends StatefulWidget {
  const QuizSetupPage({required this.preferredLanguage, super.key});

  final String preferredLanguage;

  @override
  State<QuizSetupPage> createState() => _QuizSetupPageState();
}

class _QuizSetupPageState extends State<QuizSetupPage> {
  final QuizApi _quizApi = QuizApi();
  final VocabularyApi _vocabularyApi = VocabularyApi();

  List<CurriculumCategory> _categories = [];

  String? _selectedCategoryId;
  int _questionCount = 5;
  int? _difficulty;

  bool _loadingCategories = true;
  bool _startingQuiz = false;
  String? _error;

  @override
  void initState() {
    super.initState();

    _loadCategories();
  }

  @override
  void dispose() {
    _quizApi.dispose();
    _vocabularyApi.dispose();
    super.dispose();
  }

  Future<void> _loadCategories() async {
    try {
      final categories = await _vocabularyApi.fetchCategories();

      if (!mounted) {
        return;
      }

      setState(() {
        _categories = categories;
        _loadingCategories = false;
      });
    } on ApiClientException catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _error = error.message;
        _loadingCategories = false;
      });
    }
  }

  Future<void> _startQuiz() async {
    setState(() {
      _startingQuiz = true;
      _error = null;
    });

    try {
      final QuizSession session = await _quizApi.createSession(
        promptLanguage: widget.preferredLanguage,
        questionCount: _questionCount,
        categoryId: _selectedCategoryId,
        difficulty: _difficulty,
      );

      if (!mounted) {
        return;
      }

      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (context) {
            return QuizPage(initialSession: session);
          },
        ),
      );

      if (mounted) {
        setState(() {
          _startingQuiz = false;
        });
      }
    } on ApiClientException catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _error = error.message;
        _startingQuiz = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Watch each sign-language video '
          'and select the correct meaning.',
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            Text(
              'Adaptive Quiz Setup',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 8),
            const Text(
              'Receptive mode is enabled. '
              'Productive mode will be enabled '
              'after the Objective 2 recognition '
              'adapter is connected.',
            ),
            const SizedBox(height: 24),
            if (_loadingCategories)
              const Center(child: CircularProgressIndicator())
            else
              DropdownButtonFormField<String?>(
                value: _selectedCategoryId,
                decoration: const InputDecoration(
                  labelText: 'Category',
                  border: OutlineInputBorder(),
                ),
                items: [
                  const DropdownMenuItem<String?>(
                    value: null,
                    child: Text('All categories'),
                  ),
                  ..._categories.map((CurriculumCategory category) {
                    return DropdownMenuItem<String?>(
                      value: category.id,
                      child: Text(
                        category.name.forLanguage(widget.preferredLanguage),
                      ),
                    );
                  }),
                ],
                onChanged: (value) {
                  setState(() {
                    _selectedCategoryId = value;
                  });
                },
              ),
            const SizedBox(height: 16),
            DropdownButtonFormField<int>(
              value: _questionCount,
              decoration: const InputDecoration(
                labelText: 'Number of questions',
                border: OutlineInputBorder(),
              ),
              items: const [
                DropdownMenuItem(value: 1, child: Text('1 question')),
                DropdownMenuItem(value: 3, child: Text('3 questions')),
                DropdownMenuItem(value: 5, child: Text('5 questions')),
              ],
              onChanged: (value) {
                if (value != null) {
                  setState(() {
                    _questionCount = value;
                  });
                }
              },
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<int?>(
              value: _difficulty,
              decoration: const InputDecoration(
                labelText: 'Difficulty',
                border: OutlineInputBorder(),
              ),
              items: const [
                DropdownMenuItem<int?>(
                  value: null,
                  child: Text('All difficulties'),
                ),
                DropdownMenuItem<int?>(value: 1, child: Text('Difficulty 1')),
                DropdownMenuItem<int?>(value: 2, child: Text('Difficulty 2')),
                DropdownMenuItem<int?>(value: 3, child: Text('Difficulty 3')),
              ],
              onChanged: (value) {
                setState(() {
                  _difficulty = value;
                });
              },
            ),
            const SizedBox(height: 24),
            if (_error != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(_error!, textAlign: TextAlign.center),
                ),
              ),
            FilledButton.icon(
              onPressed: _startingQuiz ? null : _startQuiz,
              icon: const Icon(Icons.play_arrow),
              label: _startingQuiz
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Start Receptive Quiz'),
            ),
          ],
        ),
      ),
    );
  }
}