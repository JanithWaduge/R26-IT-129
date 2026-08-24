// lib/screens/teacher_dashboard_screen_hansika.dart
import 'package:flutter/material.dart';
import '../constants.dart';
import '../services/teacher_api_service.dart';
import '../services/teacher_session.dart';
import 'add_sign_screen_hansika.dart';
import 'my_submissions_screen_hansika.dart';
import 'send_to_authority_screen_hansika.dart';

class TeacherDashboardScreenHansika extends StatefulWidget {
  const TeacherDashboardScreenHansika({super.key});
  @override
  State<TeacherDashboardScreenHansika> createState() =>
      _TeacherDashboardScreenHansikaState();
}

class _TeacherDashboardScreenHansikaState
    extends State<TeacherDashboardScreenHansika> {
  bool _serverOnline = false;
  List<Map<String, dynamic>> _vocabulary = []; // newly added (teacher-approved) signs
  List<String> _classroomSigns = []; // NEW: original 30 classroom signs, read-only
  bool _loading = true;

  List<Map<String, dynamic>> _mySubmissions = [];
  String _searchQuery = '';
  String _categoryFilter = 'all';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _ensureTeacherId());
    _loadData();
  }

  Future<void> _ensureTeacherId() async {
    if (TeacherSession.teacherId != null && TeacherSession.teacherEmail != null) return;
    final nameController = TextEditingController();
    final emailController = TextEditingController();
    await showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => StatefulBuilder(
        builder: (context, setDialogState) {
          String? error;
          return AlertDialog(
            backgroundColor: Colors.white,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
            title: const Text('Teacher Details', style: TextStyle(color: kInk, fontWeight: FontWeight.w700)),
            content: Column(mainAxisSize: MainAxisSize.min, children: [
              TextField(
                controller: nameController,
                autofocus: true,
                style: const TextStyle(color: kInk),
                decoration: InputDecoration(
                  hintText: 'Your name',
                  hintStyle: const TextStyle(color: kInkSoft),
                  filled: true,
                  fillColor: kSurface,
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: emailController,
                keyboardType: TextInputType.emailAddress,
                style: const TextStyle(color: kInk),
                decoration: InputDecoration(
                  hintText: 'Your email (for approval updates)',
                  hintStyle: const TextStyle(color: kInkSoft),
                  filled: true,
                  fillColor: kSurface,
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                ),
              ),
              if (error != null) ...[
                const SizedBox(height: 8),
                Text(error, style: const TextStyle(color: kError, fontSize: 12)),
              ],
            ]),
            actions: [
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: kPrimary,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  onPressed: () {
                    final email = emailController.text.trim();
                    if (!email.contains('@') || !email.contains('.')) {
                      setDialogState(() => error = 'Please enter a valid email address.');
                      return;
                    }
                    TeacherSession.teacherId = nameController.text.trim().isEmpty
                        ? 'teacher_${DateTime.now().millisecondsSinceEpoch}'
                        : nameController.text.trim();
                    TeacherSession.teacherEmail = email;
                    Navigator.pop(context);
                    _loadData();
                  },
                  child: const Text('Continue'),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Future<void> _loadData() async {
    if (mounted) setState(() => _loading = true);
    final online = await TeacherApiService.checkHealth();
    final vocab = await TeacherApiService.getVocabulary();
    final classroomSigns = await TeacherApiService.getClassroomSigns(); // NEW
    final teacherId = TeacherSession.teacherId;
    final mine = teacherId != null
        ? await TeacherApiService.getMySubmissions(teacherId)
        : <Map<String, dynamic>>[];
    if (!mounted) return;
    setState(() {
      _serverOnline = online;
      _vocabulary = vocab;
      _classroomSigns = classroomSigns;
      _mySubmissions = mine;
      _loading = false;
    });
  }

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
              _buildWelcomeBanner(),
              const SizedBox(height: 20),
              _buildServerStatus(),
              const SizedBox(height: 20),
              _buildStatsGrid(),
              const SizedBox(height: 24),
              _sectionLabel('QUICK ACTIONS'),
              const SizedBox(height: 12),
              _buildActionButtons(context),
              const SizedBox(height: 12),
              _buildSendToAuthorityCard(context),
              const SizedBox(height: 28),
              _buildRecentActivitySection(context),
              const SizedBox(height: 28),
              // ── NEW: classroom signs section (original 30, read-only) ──
              _buildClassroomSignsSection(),
              const SizedBox(height: 28),
              // ── RENAMED: was "SIGN VOCABULARY", now clearly separated ──
              _sectionLabel('NEWLY ADDED SIGNS'),
              const SizedBox(height: 12),
              _buildVocabularyFilters(),
              const SizedBox(height: 14),
              _buildVocabularySection(),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Row(children: [
      IconButton(
        icon: const Icon(Icons.arrow_back_ios_rounded, color: kInkSoft, size: 20),
        onPressed: () => Navigator.maybePop(context),
      ),
      const SizedBox(width: 4),
      const Expanded(
        child: Text('Teacher Dashboard',
            style: TextStyle(color: kInk, fontSize: 20, fontWeight: FontWeight.w800)),
      ),
    ]);
  }

  Widget _buildWelcomeBanner() {
    final name = TeacherSession.teacherId ?? 'Teacher';
    final hour = DateTime.now().hour;
    final greeting = hour < 12 ? 'Good morning' : (hour < 17 ? 'Good afternoon' : 'Good evening');
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft, end: Alignment.bottomRight,
          colors: [kPrimary, kSecondary],
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(color: kPrimary.withOpacity(0.28), blurRadius: 26, offset: const Offset(0, 12)),
        ],
      ),
      child: Row(children: [
        Container(
          width: 52, height: 52,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.18),
            borderRadius: BorderRadius.circular(16),
          ),
          child: const Icon(Icons.school_rounded, color: Colors.white, size: 26),
        ),
        const SizedBox(width: 14),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('$greeting,', style: const TextStyle(color: Colors.white70, fontSize: 12)),
          Text(name, style: const TextStyle(color: Colors.white, fontSize: 17, fontWeight: FontWeight.w700)),
        ])),
      ]),
    );
  }

  Widget _buildServerStatus() {
    final color = _serverOnline ? kSuccess : kError;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.25)),
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
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFEDEFF5)),
        boxShadow: [
          BoxShadow(color: color.withOpacity(0.12), blurRadius: 16, offset: const Offset(0, 8)),
        ],
      ),
      child: Column(children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(height: 8),
        Text(value, style: TextStyle(color: color, fontSize: 20, fontWeight: FontWeight.w800)),
        const SizedBox(height: 2),
        Text(label, style: const TextStyle(color: kInkSoft, fontSize: 11)),
      ]),
    );
  }

  Widget _sectionLabel(String text) => Text(text,
      style: const TextStyle(color: kInkSoft, fontSize: 11, fontWeight: FontWeight.w700, letterSpacing: 1.5));

  Widget _buildActionButtons(BuildContext context) {
    return Row(children: [
      Expanded(child: _actionCard(
        icon: Icons.add_circle_outline_rounded,
        label: 'Add New Sign',
        color: kSecondary,
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

  Widget _buildSendToAuthorityCard(BuildContext context) {
    return GestureDetector(
      onTap: () => Navigator.push(context,
          MaterialPageRoute(builder: (_) => const SendToAuthorityScreenHansika())),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
        decoration: BoxDecoration(
          color: kSuccess.withOpacity(0.08),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: kSuccess.withOpacity(0.22)),
        ),
        child: Row(children: [
          Container(
            width: 34, height: 34,
            decoration: BoxDecoration(color: kSuccess.withOpacity(0.16), shape: BoxShape.circle),
            child: const Icon(Icons.send_rounded, color: kSuccess, size: 17),
          ),
          const SizedBox(width: 12),
          // ── CHANGED: label no longer implies a fixed "batch" size ──
          const Expanded(child: Text('Send Pending Signs to Authority',
              style: TextStyle(color: kSuccess, fontWeight: FontWeight.w700, fontSize: 13))),
          const Icon(Icons.arrow_forward_ios_rounded, color: kSuccess, size: 14),
        ]),
      ),
    );
  }

  Widget _actionCard({required IconData icon, required String label, required Color color, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: const Color(0xFFEDEFF5)),
          boxShadow: [
            BoxShadow(color: color.withOpacity(0.14), blurRadius: 18, offset: const Offset(0, 8)),
          ],
        ),
        child: Column(children: [
          Container(
            width: 40, height: 40,
            decoration: BoxDecoration(
              gradient: LinearGradient(colors: [color, color.withOpacity(0.7)]),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: Colors.white, size: 20),
          ),
          const SizedBox(height: 10),
          Text(label, textAlign: TextAlign.center,
              style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 13)),
        ]),
      ),
    );
  }

  Widget _buildRecentActivitySection(BuildContext context) {
    final recent = _mySubmissions.take(3).toList();
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Expanded(child: _sectionLabel('RECENT ACTIVITY')),
        GestureDetector(
          onTap: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const MySubmissionsScreenHansika())),
          child: Row(children: [
            Text('View All', style: TextStyle(color: kPrimary.withOpacity(0.85), fontSize: 12, fontWeight: FontWeight.w600)),
            const SizedBox(width: 2),
            Icon(Icons.arrow_forward_ios_rounded, size: 10, color: kPrimary.withOpacity(0.85)),
          ]),
        ),
      ]),
      const SizedBox(height: 12),
      if (_loading)
        const Center(child: Padding(padding: EdgeInsets.all(16), child: CircularProgressIndicator(color: kPrimary)))
      else if (recent.isEmpty)
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(color: kSurface, borderRadius: BorderRadius.circular(14)),
          child: const Text('No submissions yet — add your first sign above.',
              style: TextStyle(color: kInkSoft, fontSize: 12)),
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
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xFFEDEFF5)),
              boxShadow: [BoxShadow(color: color.withOpacity(0.08), blurRadius: 12, offset: const Offset(0, 6))],
            ),
            child: Row(children: [
              Icon(icon, color: color, size: 18),
              const SizedBox(width: 10),
              Expanded(child: Text(s['english_word'] ?? '',
                  style: const TextStyle(color: kInk, fontSize: 13, fontWeight: FontWeight.w600))),
              Text(status.toString().toUpperCase(),
                  style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.w700)),
            ]),
          );
        }).toList()),
    ]);
  }

  // ════════════════════════════════════════════
  // NEW — Classroom Signs (original 30, read-only reference list)
  // ════════════════════════════════════════════
  Widget _buildClassroomSignsSection() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sectionLabel('CLASSROOM SIGNS (${_classroomSigns.length})'),
      const SizedBox(height: 4),
      Text('The original sign set the recognition model was trained on.',
          style: TextStyle(color: kInkSoft.withOpacity(0.8), fontSize: 11)),
      const SizedBox(height: 12),
      if (_loading)
        const Center(child: Padding(padding: EdgeInsets.all(16), child: CircularProgressIndicator(color: kPrimary)))
      else if (_classroomSigns.isEmpty)
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(color: kSurface, borderRadius: BorderRadius.circular(14)),
          child: const Text('Classroom sign list not available right now.',
              style: TextStyle(color: kInkSoft, fontSize: 12)),
        )
      else
        Wrap(
          spacing: 8, runSpacing: 8,
          children: _classroomSigns.map((word) => Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: kInk.withOpacity(0.05),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: kInk.withOpacity(0.12)),
            ),
            child: Text(word,
                style: const TextStyle(color: kInk, fontSize: 12, fontWeight: FontWeight.w600)),
          )).toList(),
        ),
    ]);
  }

  Widget _buildVocabularyFilters() {
    return Column(children: [
      TextField(
        onChanged: (v) => setState(() => _searchQuery = v),
        style: const TextStyle(color: kInk, fontSize: 13),
        decoration: InputDecoration(
          hintText: 'Search signs...',
          hintStyle: const TextStyle(color: kInkSoft, fontSize: 13),
          prefixIcon: const Icon(Icons.search_rounded, color: kInkSoft, size: 20),
          filled: true,
          fillColor: kSurface,
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
          color: selected ? kPrimary.withOpacity(0.12) : kSurface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: selected ? kPrimary : const Color(0xFFEDEFF5)),
        ),
        child: Text(label, style: TextStyle(
            color: selected ? kPrimary : kInkSoft,
            fontSize: 12, fontWeight: FontWeight.w600)),
      ),
    );
  }

  Widget _buildVocabularySection() {
    final filtered = _filteredVocabulary;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      if (_loading)
        const Center(child: Padding(padding: EdgeInsets.all(20),
            child: CircularProgressIndicator(color: kPrimary)))
      else if (filtered.isEmpty)
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(color: kSurface, borderRadius: BorderRadius.circular(14)),
          child: Text(
              _vocabulary.isEmpty
                  ? 'No teacher-submitted signs approved yet.'
                  : 'No signs match your search.',
              style: const TextStyle(color: kInkSoft, fontSize: 12)),
        )
      else
        Wrap(
          spacing: 8, runSpacing: 8,
          children: filtered.map((v) => Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: kPrimary.withOpacity(0.08),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: kPrimary.withOpacity(0.2)),
            ),
            child: Text('${v['english_word']} · ${v['sinhala_word'] ?? ''}',
                style: const TextStyle(color: kPrimary, fontSize: 12,
                    fontWeight: FontWeight.w600)),
          )).toList(),
        ),
    ]);
  }
}