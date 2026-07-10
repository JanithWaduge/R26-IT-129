// lib/screens/teacher_dashboard_screen_hansika.dart
import 'package:flutter/material.dart';
import '../constants.dart';
import '../services/teacher_api_service.dart';
import '../services/teacher_session.dart';
import 'add_sign_screen_hansika.dart';
import 'my_submissions_screen_hansika.dart';

class TeacherDashboardScreenHansika extends StatefulWidget {
  const TeacherDashboardScreenHansika({super.key});
  @override
  State<TeacherDashboardScreenHansika> createState() =>
      _TeacherDashboardScreenHansikaState();
}

class _TeacherDashboardScreenHansikaState
    extends State<TeacherDashboardScreenHansika> {
  bool _serverOnline = false;
  List<Map<String, dynamic>> _vocabulary = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _ensureTeacherId());
    _loadData();
  }

  Future<void> _ensureTeacherId() async {
    if (TeacherSession.teacherId != null) return;
    final controller = TextEditingController();
    await showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        backgroundColor: kSurface,
        title: const Text('Teacher Name', style: TextStyle(color: Colors.white)),
        content: TextField(
          controller: controller,
          autofocus: true,
          style: const TextStyle(color: Colors.white),
          decoration: const InputDecoration(
            hintText: 'Enter your name',
            hintStyle: TextStyle(color: Colors.white38),
          ),
        ),
        actions: [
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: kPrimary),
            onPressed: () {
              TeacherSession.teacherId = controller.text.trim().isEmpty
                  ? 'teacher_${DateTime.now().millisecondsSinceEpoch}'
                  : controller.text.trim();
              Navigator.pop(context);
            },
            child: const Text('Continue'),
          ),
        ],
      ),
    );
  }

  Future<void> _loadData() async {
    final online = await TeacherApiService.checkHealth();
    final vocab = await TeacherApiService.getVocabulary();
    if (!mounted) return;
    setState(() {
      _serverOnline = online;
      _vocabulary = vocab;
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBackground,
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _loadData,
          color: kPrimary,
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              _buildHeader(context),
              const SizedBox(height: 20),
              _buildServerStatus(),
              const SizedBox(height: 24),
              _buildActionButtons(context),
              const SizedBox(height: 28),
              _buildVocabularySection(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Row(children: [
      IconButton(
        icon: const Icon(Icons.arrow_back_ios_rounded, color: Colors.white54, size: 20),
        onPressed: () => Navigator.maybePop(context),
      ),
      const SizedBox(width: 4),
      const Expanded(
        child: Text('Teacher Dashboard',
            style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w800)),
      ),
    ]);
  }

  Widget _buildServerStatus() {
    final color = _serverOnline ? kSuccess : kError;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(children: [
        Icon(_serverOnline ? Icons.cloud_done_rounded : Icons.cloud_off_rounded,
            color: color, size: 18),
        const SizedBox(width: 10),
        Text(_serverOnline ? 'Server connected' : 'Server offline — check PC server',
            style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600)),
      ]),
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    return Column(children: [
      SizedBox(
        width: double.infinity,
        height: 56,
        child: ElevatedButton(
          onPressed: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const AddSignScreenHansika())),
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFFFFB703),
            foregroundColor: Colors.black,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          ),
          child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Icon(Icons.add_circle_outline_rounded),
            SizedBox(width: 10),
            Text('Add New Sign', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
          ]),
        ),
      ),
      const SizedBox(height: 12),
      SizedBox(
        width: double.infinity,
        height: 56,
        child: OutlinedButton(
          onPressed: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const MySubmissionsScreenHansika())),
          style: OutlinedButton.styleFrom(
            side: const BorderSide(color: Colors.white24),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          ),
          child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Icon(Icons.pending_actions_rounded, color: Colors.white),
            SizedBox(width: 10),
            Text('My Submissions',
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 15)),
          ]),
        ),
      ),
    ]);
  }

  Widget _buildVocabularySection() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('APPROVED SIGNS (${_vocabulary.length})',
          style: TextStyle(color: Colors.white.withOpacity(0.35), fontSize: 11,
              fontWeight: FontWeight.w700, letterSpacing: 1.5)),
      const SizedBox(height: 12),
      if (_loading)
        const Center(child: Padding(padding: EdgeInsets.all(20),
            child: CircularProgressIndicator(color: kPrimary)))
      else if (_vocabulary.isEmpty)
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.03),
              borderRadius: BorderRadius.circular(14)),
          child: Text('No teacher-submitted signs approved yet.',
              style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 12)),
        )
      else
        Wrap(
          spacing: 8, runSpacing: 8,
          children: _vocabulary.map((v) => Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: const Color(0xFFFFB703).withOpacity(0.08),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: const Color(0xFFFFB703).withOpacity(0.25)),
            ),
            child: Text('${v['english_word']} · ${v['sinhala_word'] ?? ''}',
                style: const TextStyle(color: Color(0xFFFFB703), fontSize: 12,
                    fontWeight: FontWeight.w600)),
          )).toList(),
        ),
    ]);
  }
}