// lib/screens/my_submissions_screen_hansika.dart
import 'package:flutter/material.dart';
import '../constants.dart';
import '../services/teacher_api_service.dart';
import '../services/teacher_session.dart';

class MySubmissionsScreenHansika extends StatefulWidget {
  const MySubmissionsScreenHansika({super.key});
  @override
  State<MySubmissionsScreenHansika> createState() => _MySubmissionsScreenHansikaState();
}

class _MySubmissionsScreenHansikaState extends State<MySubmissionsScreenHansika> {
  List<Map<String, dynamic>> _submissions = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final teacherId = TeacherSession.teacherId ?? 'unknown_teacher';
    final data = await TeacherApiService.getMySubmissions(teacherId);
    if (!mounted) return;
    setState(() { _submissions = data; _loading = false; });
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'approved': return kSuccess;
      case 'rejected': return kError;
      default: return kWarning;
    }
  }

  IconData _statusIcon(String status) {
    switch (status) {
      case 'approved': return Icons.check_circle_rounded;
      case 'rejected': return Icons.cancel_rounded;
      default: return Icons.hourglass_top_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBackground,
      appBar: AppBar(
        backgroundColor: Colors.transparent, elevation: 0,
        title: const Text('My Submissions', style: TextStyle(color: Colors.white)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _load,
          color: kPrimary,
          child: _loading
              ? const Center(child: CircularProgressIndicator(color: kPrimary))
              : _submissions.isEmpty
                  ? ListView(children: [
                      Padding(
                        padding: const EdgeInsets.all(40),
                        child: Column(children: [
                          Icon(Icons.inbox_rounded, color: Colors.white.withOpacity(0.2), size: 56),
                          const SizedBox(height: 16),
                          Text('No submissions yet', style: TextStyle(color: Colors.white.withOpacity(0.4))),
                        ]),
                      ),
                    ])
                  : ListView.builder(
                      padding: const EdgeInsets.all(16),
                      itemCount: _submissions.length,
                      itemBuilder: (context, i) {
                        final s = _submissions[i];
                        final status = s['status'] ?? 'pending';
                        final color = _statusColor(status);
                        return Container(
                          margin: const EdgeInsets.only(bottom: 12),
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: color.withOpacity(0.06),
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(color: color.withOpacity(0.25)),
                          ),
                          child: Row(children: [
                            Icon(_statusIcon(status), color: color, size: 26),
                            const SizedBox(width: 14),
                            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                              Text(s['english_word'] ?? '',
                                  style: const TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.w700)),
                              const SizedBox(height: 2),
                              Text(s['sinhala_word'] ?? '',
                                  style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 12)),
                              if (status == 'rejected' && s['rejection_reason'] != null) ...[
                                const SizedBox(height: 6),
                                Text('Reason: ${s['rejection_reason']}',
                                    style: TextStyle(color: kError.withOpacity(0.8), fontSize: 11)),
                              ],
                            ])),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                              decoration: BoxDecoration(color: color.withOpacity(0.15), borderRadius: BorderRadius.circular(8)),
                              child: Text(status.toString().toUpperCase(),
                                  style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.w700)),
                            ),
                          ]),
                        );
                      },
                    ),
        ),
      ),
    );
  }
}