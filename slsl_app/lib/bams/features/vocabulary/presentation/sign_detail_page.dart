import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../../../shared/widgets/sign_video_player.dart';
import '../data/vocabulary_api.dart';
import '../domain/curriculum_competency.dart';
import '../domain/sign_detail.dart';

class SignDetailPage extends StatefulWidget {
  const SignDetailPage({
    required this.signId,
    required this.preferredLanguage,
    super.key,
  });

  final String signId;
  final String preferredLanguage;

  @override
  State<SignDetailPage> createState() => _SignDetailPageState();
}

class _SignDetailPageState extends State<SignDetailPage> {
  final VocabularyApi _api = VocabularyApi();

  SignDetail? _sign;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();

    _loadSign();
  }

  @override
  void dispose() {
    _api.dispose();
    super.dispose();
  }

  Future<void> _loadSign() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final SignDetail sign = await _api.fetchSign(widget.signId);

      if (!mounted) {
        return;
      }

      setState(() {
        _sign = sign;
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Sign Details')),
      body: SafeArea(child: _buildBody(context)),
    );
  }

  Widget _buildBody(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, size: 64),
              const SizedBox(height: 16),
              Text(_error!, textAlign: TextAlign.center),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: _loadSign,
                child: const Text('Try Again'),
              ),
            ],
          ),
        ),
      );
    }

    final SignDetail sign = _sign!;

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        // FIXED: this used to be a static Icon + "Media reference available"
        // text with no actual playback behind it. SignVideoPlayer already
        // exists in the codebase (and is used correctly in the quiz
        // screens) - it just was never wired in here. It manages its own
        // sizing/border via AspectRatio, so no outer Container needed.
        sign.media.isAvailable
            ? SignVideoPlayer(mediaUri: sign.media.uri!)
            : Container(
                height: 220,
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
                        'Validated sign media '
                        'has not been attached.',
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
        const SizedBox(height: 24),
        Text(
          sign.meanings.forLanguage(widget.preferredLanguage),
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 6),
        Text('SLSL gloss: ${sign.gloss}'),
        const SizedBox(height: 20),
        _DetailCard(
          title: 'Meanings',
          children: [
            _DetailRow(label: 'English', value: sign.meanings.english),
            _DetailRow(label: 'Sinhala', value: sign.meanings.sinhala),
            _DetailRow(label: 'Tamil', value: sign.meanings.tamil),
          ],
        ),
        _DetailCard(
          title: 'Classification',
          children: [
            _DetailRow(
              label: 'Category',
              value: sign.category.name.forLanguage(widget.preferredLanguage),
            ),
            _DetailRow(label: 'Difficulty', value: '${sign.difficulty} of 5'),
            _DetailRow(
              label: 'Validation status',
              value: sign.validationStatus,
            ),
            _DetailRow(label: 'Content status', value: sign.contentStatus),
          ],
        ),
        _DetailCard(
          title: 'Competencies',
          children: sign.competencies.isEmpty
              ? const [Text('No competency is linked.')]
              : sign.competencies.map((CurriculumCompetency competency) {
                  return ListTile(
                    contentPadding: EdgeInsets.zero,
                    leading: const Icon(Icons.checklist),
                    title: Text(
                      competency.title.forLanguage(widget.preferredLanguage),
                    ),
                    subtitle: Text(
                      'Receptive target: '
                      '${(competency.receptiveThreshold * 100).round()}%\n'
                      'Productive target: '
                      '${(competency.productiveThreshold * 100).round()}%',
                    ),
                  );
                }).toList(),
        ),
        _DetailCard(
          title: 'Prerequisites',
          children: sign.prerequisites.isEmpty
              ? const [Text('No prerequisite signs.')]
              : sign.prerequisites.map((prerequisite) {
                  return ListTile(
                    contentPadding: EdgeInsets.zero,
                    leading: const Icon(Icons.account_tree),
                    title: Text(
                      prerequisite.meanings.forLanguage(
                        widget.preferredLanguage,
                      ),
                    ),
                    subtitle: Text(prerequisite.gloss),
                  );
                }).toList(),
        ),
        _DetailCard(
          title: 'Media integration',
          children: [
            _DetailRow(label: 'Source type', value: sign.media.sourceType),
            _DetailRow(
              label: 'Objective source',
              value: sign.media.objectiveSource,
            ),
            _DetailRow(
              label: 'Status',
              value: sign.media.isAvailable ? 'Available' : 'Pending',
            ),
          ],
        ),
        if (sign.validationStatus == 'provisional')
          Card(
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.info_outline),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'This is development '
                      'content. It must not be '
                      'used as validated SLSL '
                      'curriculum material until '
                      'a qualified teacher approves it.',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }
}

class _DetailCard extends StatelessWidget {
  const _DetailCard({required this.title, required this.children});

  final String title;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            ...children,
          ],
        ),
      ),
    );
  }
}

class _DetailRow extends StatelessWidget {
  const _DetailRow({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Text(
              label,
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(flex: 2, child: Text(value)),
        ],
      ),
    );
  }
}