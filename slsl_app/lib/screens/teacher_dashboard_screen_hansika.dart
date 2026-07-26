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

  // ── NEW: stats + recent activity + vocab filter state ──
  List<Map<String, dynamic>> _mySubmissions = [];
  String _searchQuery = '';
  String _categoryFilter = 'all'; // all | noun | verb

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
              _loadData(); // refresh with the new teacher id
            },
            child: const Text('Continue'),
          ),
        ],
      ),
    );
  }

  Future<void> _loadData() async {
    if (mounted) setState(() => _loading = true);
    final online = await TeacherApiService.checkHealth();
    final vocab = await TeacherApiService.getVocabulary();
    final teacherId = TeacherSession.teacherId;
    final mine = teacherId != null
        ? await TeacherApiService.getMySubmissions(teacherId)
        : <Map<String, dynamic>>[];
    if (!mounted) return;
    setState(() {
      _serverOnline = online;
      _vocabulary = vocab;
      _mySubmissions = mine;
      _loading = false;
    });
  }

  // ── NEW: computed stats from submissions ──
  int get _pendingCount => _mySubmissions.where((s) => s['status'] == 'pending').length;
  int get _approvedCount => _mySubmissions.where((s) => s['status'] == 'approved').length;
  int get _rejectedCount => _mySubmissions.where((s) => s['status'] == 'rejected').length;

  List<Map<String, dynamic>> get _filteredVocabulary {
    return _vocabulary.where((v) {
      final matchesSearch = _searchQuery.isEmpty ||
          (v['english_word'] ?? '').toString().toLowerCase().contains(_searchQuery.toLowerCase()) ||
          (v['sinhala_word'] ?? '').toString().toLowerCase().contains(_searchQuery.toLowerCase());
      final matchesCategory = _categoryFilter == 'all' || v['category'] == _categoryFilter;
      return matchesSearch && matchesCategory;
    }).toList();
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
              const SizedBox(height: 8),
              _buildWelcomeBanner(),           // NEW
              const SizedBox(height: 20),
              _buildServerStatus(),
              const SizedBox(height: 20),
              _buildStatsGrid(),                // NEW
              const SizedBox(height: 24),
              _sectionLabel('QUICK ACTIONS'),   // NEW label wrapper
              const SizedBox(height: 12),
              _buildActionButtons(context),
              const SizedBox(height: 28),
              _buildRecentActivitySection(context), // NEW
              const SizedBox(height: 28),
              _sectionLabel('SIGN VOCABULARY'), // NEW label wrapper
              const SizedBox(height: 12),
              _buildVocabularyFilters(),         // NEW
              const SizedBox(height: 14),
              _buildVocabularySection(),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  // ════════════════════════════════════════════
  // HEADER (unchanged logic, kept as-is)
  // ════════════════════════════════════════════
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

  // ════════════════════════════════════════════
  // NEW — WELCOME BANNER
  // ════════════════════════════════════════════
  Widget _buildWelcomeBanner() {
    final name = TeacherSession.teacherId ?? 'Teacher';
    final hour = DateTime.now().hour;
    final greeting = hour < 12 ? 'Good morning' : (hour < 17 ? 'Good afternoon' : 'Good evening');
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft, end: Alignment.bottomRight,
          colors: [const Color(0xFFFFB703).withOpacity(0.18), const Color(0xFFFB8500).withOpacity(0.08)],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFFFB703).withOpacity(0.25)),
      ),
      child: Row(children: [
        Container(
          width: 52, height: 52,
          decoration: BoxDecoration(
            color: const Color(0xFFFFB703).withOpacity(0.15),
            borderRadius: BorderRadius.circular(16),
          ),
          child: const Icon(Icons.school_rounded, color: Color(0xFFFFB703), size: 26),
        ),
        const SizedBox(width: 14),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('$greeting,', style: TextStyle(color: Colors.white.withOpacity(0.6), fontSize: 12)),
          Text(name, style: const TextStyle(color: Colors.white, fontSize: 17, fontWeight: FontWeight.w700)),
        ])),
      ]),
    );
  }

  // ════════════════════════════════════════════
  // SERVER STATUS (unchanged)
  // ════════════════════════════════════════════
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

  // ════════════════════════════════════════════
  // NEW — STATS GRID
  // ════════════════════════════════════════════
  Widget _buildStatsGrid() {
    return Row(children: [
      Expanded(child: _statCard('Approved', _vocabulary.length.toString(), Icons.check_circle_rounded, kSuccess)),
      const SizedBox(width: 10),
      Expanded(child: _statCard('Pending', _pendingCount.toString(), Icons.hourglass_top_rounded, kWarning)),
      const SizedBox(width: 10),
      Expanded(child: _statCard('Rejected', _rejectedCount.toString(), Icons.cancel_rounded, kError)),
    ]);
  }

  Widget _statCard(String label, String value, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.07),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Column(children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(height: 8),
        Text(value, style: TextStyle(color: color, fontSize: 20, fontWeight: FontWeight.w800)),
        const SizedBox(height: 2),
        Text(label, style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 11)),
      ]),
    );
  }

  Widget _sectionLabel(String text) => Text(text,
      style: TextStyle(color: Colors.white.withOpacity(0.35), fontSize: 11,
          fontWeight: FontWeight.w700, letterSpacing: 1.5));

  // ════════════════════════════════════════════
  // ACTION BUTTONS (unchanged logic — restyled as side-by-side cards)
  // ════════════════════════════════════════════
  Widget _buildActionButtons(BuildContext context) {
    return Row(children: [
      Expanded(child: _actionCard(
        icon: Icons.add_circle_outline_rounded,
        label: 'Add New Sign',
        color: const Color(0xFFFFB703),
        onTap: () => Navigator.push(context,
            MaterialPageRoute(builder: (_) => const AddSignScreenHansika())),
      )),
      const SizedBox(width: 12),
      Expanded(child: _actionCard(
        icon: Icons.pending_actions_rounded,
        label: 'My Submissions',
        color: kPrimary,
        onTap: () => Navigator.push(context,
            MaterialPageRoute(builder: (_) => const MySubmissionsScreenHansika())),
      )),
    ]);
  }

  Widget _actionCard({required IconData icon, required String label, required Color color, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 12),
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Column(children: [
          Icon(icon, color: color, size: 26),
          const SizedBox(height: 10),
          Text(label, textAlign: TextAlign.center,
              style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 13)),
        ]),
      ),
    );
  }

  // ════════════════════════════════════════════
  // NEW — RECENT ACTIVITY (last 3 submissions)
  // ════════════════════════════════════════════
  Widget _buildRecentActivitySection(BuildContext context) {
    final recent = _mySubmissions.take(3).toList();
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Expanded(child: _sectionLabel('RECENT ACTIVITY')),
        GestureDetector(
          onTap: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const MySubmissionsScreenHansika())),
          child: Row(children: [
            Text('View All', style: TextStyle(color: kPrimary.withOpacity(0.8), fontSize: 12, fontWeight: FontWeight.w600)),
            const SizedBox(width: 2),
            Icon(Icons.arrow_forward_ios_rounded, size: 10, color: kPrimary.withOpacity(0.8)),
          ]),
        ),
      ]),
      const SizedBox(height: 12),
      if (_loading)
        const Center(child: Padding(padding: EdgeInsets.all(16), child: CircularProgressIndicator(color: kPrimary)))
      else if (recent.isEmpty)
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.03), borderRadius: BorderRadius.circular(14)),
          child: Text('No submissions yet — add your first sign above.',
              style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 12)),
        )
      else
        Column(children: recent.map((s) {
          final status = s['status'] ?? 'pending';
          final color = status == 'approved' ? kSuccess : (status == 'rejected' ? kError : kWarning);
          final icon = status == 'approved' ? Icons.check_circle_rounded
              : (status == 'rejected' ? Icons.cancel_rounded : Icons.hourglass_top_rounded);
          return Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            decoration: BoxDecoration(
              color: color.withOpacity(0.06),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: color.withOpacity(0.2)),
            ),
            child: Row(children: [
              Icon(icon, color: color, size: 18),
              const SizedBox(width: 10),
              Expanded(child: Text(s['english_word'] ?? '',
                  style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600))),
              Text(status.toString().toUpperCase(),
                  style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.w700)),
            ]),
          );
        }).toList()),
    ]);
  }

  // ════════════════════════════════════════════
  // NEW — VOCABULARY FILTERS (search + category chips)
  // ════════════════════════════════════════════
  Widget _buildVocabularyFilters() {
    return Column(children: [
      TextField(
        onChanged: (v) => setState(() => _searchQuery = v),
        style: const TextStyle(color: Colors.white, fontSize: 13),
        decoration: InputDecoration(
          hintText: 'Search signs...',
          hintStyle: TextStyle(color: Colors.white.withOpacity(0.35), fontSize: 13),
          prefixIcon: Icon(Icons.search_rounded, color: Colors.white.withOpacity(0.4), size: 20),
          filled: true,
          fillColor: kSurface.withOpacity(0.3),
          contentPadding: const EdgeInsets.symmetric(vertical: 12),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
        ),
      ),
      const SizedBox(height: 10),
      Row(children: [
        _filterChip('all', 'All'),
        const SizedBox(width: 8),
        _filterChip('noun', 'Nouns'),
        const SizedBox(width: 8),
        _filterChip('verb', 'Verbs'),
      ]),
    ]);
  }

  Widget _filterChip(String value, String label) {
    final selected = _categoryFilter == value;
    return GestureDetector(
      onTap: () => setState(() => _categoryFilter = value),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? const Color(0xFFFFB703).withOpacity(0.15) : Colors.white.withOpacity(0.04),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: selected ? const Color(0xFFFFB703) : Colors.white12),
        ),
        child: Text(label, style: TextStyle(
            color: selected ? const Color(0xFFFFB703) : Colors.white54,
            fontSize: 12, fontWeight: FontWeight.w600)),
      ),
    );
  }

  // ════════════════════════════════════════════
  // VOCABULARY SECTION (same data/logic, now uses _filteredVocabulary)
  // ════════════════════════════════════════════
  Widget _buildVocabularySection() {
    final filtered = _filteredVocabulary;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      if (_loading)
        const Center(child: Padding(padding: EdgeInsets.all(20),
            child: CircularProgressIndicator(color: kPrimary)))
      else if (filtered.isEmpty)
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.03),
              borderRadius: BorderRadius.circular(14)),
          child: Text(
              _vocabulary.isEmpty
                  ? 'No teacher-submitted signs approved yet.'
                  : 'No signs match your search.',
              style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 12)),
        )
      else
        Wrap(
          spacing: 8, runSpacing: 8,
          children: filtered.map((v) => Container(
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