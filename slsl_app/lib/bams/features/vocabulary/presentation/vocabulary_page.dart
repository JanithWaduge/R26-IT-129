import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../data/vocabulary_api.dart';
import '../domain/curriculum_category.dart';
import '../domain/paginated_signs.dart';
import '../domain/sign_summary.dart';
import 'sign_detail_page.dart';

class VocabularyPage extends StatefulWidget {
  const VocabularyPage({required this.preferredLanguage, super.key});

  final String preferredLanguage;

  @override
  State<VocabularyPage> createState() => _VocabularyPageState();
}

class _VocabularyPageState extends State<VocabularyPage> {
  final VocabularyApi _api = VocabularyApi();

  final TextEditingController _searchController = TextEditingController();

  List<CurriculumCategory> _categories = [];

  PaginatedSigns? _signs;

  String? _selectedCategoryId;
  int? _selectedDifficulty;

  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();

    _loadInitialData();
  }

  @override
  void dispose() {
    _searchController.dispose();
    _api.dispose();
    super.dispose();
  }

  Future<void> _loadInitialData() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final List<CurriculumCategory> categories = await _api.fetchCategories();

      final PaginatedSigns signs = await _api.fetchSigns();

      if (!mounted) {
        return;
      }

      setState(() {
        _categories = categories;
        _signs = signs;
        _loading = false;
      });
    } on ApiClientException catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _error = error.message;
        _loading = false;
      });
    } catch (_) {
      if (!mounted) {
        return;
      }

      setState(() {
        _error = 'Unable to load vocabulary.';
        _loading = false;
      });
    }
  }

  Future<void> _loadSigns() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final PaginatedSigns signs = await _api.fetchSigns(
        categoryId: _selectedCategoryId,
        difficulty: _selectedDifficulty,
        search: _searchController.text,
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _signs = signs;
        _loading = false;
      });
    } on ApiClientException catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _error = error.message;
        _loading = false;
      });
    }
  }

  Future<void> _openSign(SignSummary sign) async {
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (context) {
          return SignDetailPage(
            signId: sign.id,
            preferredLanguage: widget.preferredLanguage,
          );
        },
      ),
    );
  }

  IconData _categoryIcon(String iconKey) {
    switch (iconKey) {
      case 'waving_hand':
        return Icons.waving_hand;
      case 'school':
        return Icons.school;
      case 'numbers':
        return Icons.numbers;
      case 'mood':
        return Icons.mood;
      default:
        return Icons.sign_language;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SLSL Vocabulary')),
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _loadInitialData,
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              TextField(
                controller: _searchController,
                textInputAction: TextInputAction.search,
                onSubmitted: (_) {
                  _loadSigns();
                },
                decoration: InputDecoration(
                  labelText: 'Search vocabulary',
                  hintText: 'Search by word, gloss or tag',
                  prefixIcon: const Icon(Icons.search),
                  suffixIcon: IconButton(
                    onPressed: _loadSigns,
                    icon: const Icon(Icons.arrow_forward),
                  ),
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 20),
              Text('Categories', style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 12),
              SizedBox(
                height: 108,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  children: [
                    _CategoryCard(
                      title: 'All',
                      icon: Icons.apps,
                      selected: _selectedCategoryId == null,
                      onTap: () {
                        setState(() {
                          _selectedCategoryId = null;
                        });

                        _loadSigns();
                      },
                    ),
                    ..._categories.map((CurriculumCategory category) {
                      return _CategoryCard(
                        title: category.name.forLanguage(
                          widget.preferredLanguage,
                        ),
                        icon: _categoryIcon(category.iconKey),
                        selected: _selectedCategoryId == category.id,
                        onTap: () {
                          setState(() {
                            _selectedCategoryId = category.id;
                          });

                          _loadSigns();
                        },
                      );
                    }),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              DropdownButtonFormField<int?>(
                value: _selectedDifficulty,
                decoration: const InputDecoration(
                  labelText: 'Difficulty filter',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.tune),
                ),
                items: const [
                  DropdownMenuItem<int?>(
                    value: null,
                    child: Text('All difficulties'),
                  ),
                  DropdownMenuItem<int?>(value: 1, child: Text('Difficulty 1')),
                  DropdownMenuItem<int?>(value: 2, child: Text('Difficulty 2')),
                  DropdownMenuItem<int?>(value: 3, child: Text('Difficulty 3')),
                  DropdownMenuItem<int?>(value: 4, child: Text('Difficulty 4')),
                  DropdownMenuItem<int?>(value: 5, child: Text('Difficulty 5')),
                ],
                onChanged: (int? value) {
                  setState(() {
                    _selectedDifficulty = value;
                  });

                  _loadSigns();
                },
              ),
              const SizedBox(height: 24),
              if (_loading)
                const Padding(
                  padding: EdgeInsets.all(48),
                  child: Center(child: CircularProgressIndicator()),
                )
              else if (_error != null)
                _ErrorCard(message: _error!, onRetry: _loadInitialData)
              else if (_signs == null || _signs!.items.isEmpty)
                const Padding(
                  padding: EdgeInsets.all(48),
                  child: Column(
                    children: [
                      Icon(Icons.search_off, size: 64),
                      SizedBox(height: 16),
                      Text(
                        'No matching signs were found.',
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                )
              else ...[
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        'Signs',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                    ),
                    Text('${_signs!.totalItems} items'),
                  ],
                ),
                const SizedBox(height: 12),
                ..._signs!.items.map((SignSummary sign) {
                  return Card(
                    margin: const EdgeInsets.only(bottom: 12),
                    child: ListTile(
                      onTap: () {
                        _openSign(sign);
                      },
                      leading: CircleAvatar(
                        child: Text(sign.difficulty.toString()),
                      ),
                      title: Text(
                        sign.meanings.forLanguage(widget.preferredLanguage),
                      ),
                      subtitle: Text(
                        '${sign.gloss} • '
                        '${sign.validationStatus}',
                      ),
                      trailing: Icon(
                        sign.media.isAvailable
                            ? Icons.play_circle
                            : Icons.hourglass_empty,
                      ),
                    ),
                  );
                }),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _CategoryCard extends StatelessWidget {
  const _CategoryCard({
    required this.title,
    required this.icon,
    required this.selected,
    required this.onTap,
  });

  final String title;
  final IconData icon;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 12),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          width: 112,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(width: selected ? 2 : 1),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 30),
              const SizedBox(height: 8),
              Text(
                title,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message, required this.onRetry});

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            const Icon(Icons.error_outline, size: 52),
            const SizedBox(height: 12),
            Text(message, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh),
              label: const Text('Try Again'),
            ),
          ],
        ),
      ),
    );
  }
}