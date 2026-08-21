import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../../../core/theme/app_colors.dart';
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
    return Theme(
      data: AppTheme.light,
      child: Scaffold(
        backgroundColor: AppColors.background,
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
            Container(
              width: 64,
              height: 64,
              decoration: const BoxDecoration(
                gradient: AppColors.primaryGradient,
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.quiz, color: Colors.white, size: 32),
            ),
            const SizedBox(height: 16),
            const Text(
              'Adaptive Quiz Setup',
              style: TextStyle(color: AppColors.ink, fontSize: 22, fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 8),
            const Text(
              'Receptive mode is enabled. '
              'Productive mode will be enabled '
              'after the Objective 2 recognition '
              'adapter is connected.',
              style: TextStyle(color: AppColors.inkSoft, fontSize: 13),
            ),
            const SizedBox(height: 24),
            if (_loadingCategories)
              const Center(child: CircularProgressIndicator(color: AppColors.primary))
            else
              DropdownButtonFormField<String?>(
                value: _selectedCategoryId,
                style: const TextStyle(color: AppColors.ink, fontSize: 15),
                dropdownColor: Colors.white,
                decoration: const InputDecoration(
                  labelText: 'Category',
                  prefixIcon: Icon(Icons.category_outlined, color: AppColors.primary),
                ),
                items: [
                  const DropdownMenuItem<String?>(
                    value: null,
                    child: Text('All categories', style: TextStyle(color: AppColors.ink)),
                  ),
                  ..._categories.map((CurriculumCategory category) {
                    return DropdownMenuItem<String?>(
                      value: category.id,
                      child: Text(
                        category.name.forLanguage(widget.preferredLanguage),
                        style: const TextStyle(color: AppColors.ink),
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
              style: const TextStyle(color: AppColors.ink, fontSize: 15),
              dropdownColor: Colors.white,
              decoration: const InputDecoration(
                labelText: 'Number of questions',
                prefixIcon: Icon(Icons.format_list_numbered, color: AppColors.secondary),
              ),
              items: const [
                DropdownMenuItem(
                  value: 1,
                  child: Text('1 question', style: TextStyle(color: AppColors.ink)),
                ),
                DropdownMenuItem(
                  value: 3,
                  child: Text('3 questions', style: TextStyle(color: AppColors.ink)),
                ),
                DropdownMenuItem(
                  value: 5,
                  child: Text('5 questions', style: TextStyle(color: AppColors.ink)),
                ),
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
              style: const TextStyle(color: AppColors.ink, fontSize: 15),
              dropdownColor: Colors.white,
              decoration: const InputDecoration(
                labelText: 'Difficulty',
                prefixIcon: Icon(Icons.speed, color: AppColors.success),
              ),
              items: const [
                DropdownMenuItem<int?>(
                  value: null,
                  child: Text('All difficulties', style: TextStyle(color: AppColors.ink)),
                ),
                DropdownMenuItem<int?>(
                  value: 1,
                  child: Text('Difficulty 1', style: TextStyle(color: AppColors.ink)),
                ),
                DropdownMenuItem<int?>(
                  value: 2,
                  child: Text('Difficulty 2', style: TextStyle(color: AppColors.ink)),
                ),
                DropdownMenuItem<int?>(
                  value: 3,
                  child: Text('Difficulty 3', style: TextStyle(color: AppColors.ink)),
                ),
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
                color: AppColors.error.withOpacity(0.06),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(18),
                  side: BorderSide(color: AppColors.error.withOpacity(0.25)),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(_error!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: AppColors.error)),
                ),
              ),
            SizedBox(
              height: 54,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(14),
                  gradient: _startingQuiz ? null : AppColors.primaryGradient,
                  color: _startingQuiz ? AppColors.hairline : null,
                ),
                child: FilledButton.icon(
                  onPressed: _startingQuiz ? null : _startQuiz,
                  style: FilledButton.styleFrom(
                    backgroundColor: Colors.transparent,
                    shadowColor: Colors.transparent,
                    disabledBackgroundColor: Colors.transparent,
                  ),
                  icon: const Icon(Icons.play_arrow),
                  label: _startingQuiz
                      ? const SizedBox(
                          width: 22,
                          height: 22,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Text('Start Receptive Quiz'),
                ),
              ),
            ),
            ],
          ),
        ),
      ),
    );
  }
}