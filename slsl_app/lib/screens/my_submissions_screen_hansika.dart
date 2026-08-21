// lib/screens/my_submissions_screen_hansika.dart
import 'package:flutter/material.dart';
import '../constants.dart';
import '../services/teacher_api_service.dart';
import '../services/teacher_session.dart';
import 'sign_playback_screen_hansika.dart';

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
    setState(() {
      _submissions = data;
      _loading = false;
    });
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'approved':
        return kSuccess;
      case 'rejected':
        return kError;
      default:
        return kWarning;
    }
  }

  IconData _statusIcon(String status) {
    switch (status) {
      case 'approved':
        return Icons.check_circle_rounded;
      case 'rejected':
        return Icons.cancel_rounded;
      default:
        return Icons.hourglass_top_rounded;
    }
  }

  Future<void> _confirmDelete(String id, String word) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        title: const Text('Delete Sign?', style: TextStyle(color: kInk, fontWeight: FontWeight.w700)),
        content: Text(
          'This will permanently delete "$word". This cannot be undone.',
          style: const TextStyle(color: kInkSoft),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel', style: TextStyle(color: kInkSoft)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: kError,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      final ok = await TeacherApiService.deleteSubmission(id);
      if (!mounted) return;
      if (ok) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('"$word" deleted.'),
          backgroundColor: kInk,
        ));
        _load();
      } else {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Failed to delete. Try again.'),
          backgroundColor: kError,
        ));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBackground,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text('My Submissions', style: TextStyle(color: kInk, fontWeight: FontWeight.w700)),
        iconTheme: const IconThemeData(color: kInk),
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
                          Icon(Icons.inbox_rounded, color: kInkSoft.withOpacity(0.4), size: 56),
                          const SizedBox(height: 16),
                          const Text('No submissions yet',
                              style: TextStyle(color: kInkSoft)),
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
                        final id = s['_id'].toString();
                        final word = s['english_word'] ?? '';

                        return GestureDetector(
                          onTap: () => Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => SignPlaybackScreenHansika(
                                submissionId: id,
                                englishWord: word,
                              ),
                            ),
                          ),
                          child: Container(
                            margin: const EdgeInsets.only(bottom: 12),
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(16),
                              border: Border.all(color: const Color(0xFFEDEFF5)),
                              boxShadow: [
                                BoxShadow(color: color.withOpacity(0.10), blurRadius: 16, offset: const Offset(0, 8)),
                              ],
                            ),
                            child: Row(children: [
                              Container(
                                width: 42, height: 42,
                                decoration: BoxDecoration(color: color.withOpacity(0.14), shape: BoxShape.circle),
                                child: Icon(_statusIcon(status), color: color, size: 22),
                              ),
                              const SizedBox(width: 14),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(word,
                                        style: const TextStyle(
                                            color: kInk,
                                            fontSize: 15,
                                            fontWeight: FontWeight.w700)),
                                    const SizedBox(height: 2),
                                    Text(s['sinhala_word'] ?? '',
                                        style: const TextStyle(
                                            color: kInkSoft, fontSize: 12)),
                                    if (status == 'rejected' && s['rejection_reason'] != null) ...[
                                      const SizedBox(height: 6),
                                      Text('Reason: ${s['rejection_reason']}',
                                          style: TextStyle(
                                              color: kError.withOpacity(0.85), fontSize: 11)),
                                    ],
                                  ],
                                ),
                              ),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                                decoration: BoxDecoration(
                                    color: color.withOpacity(0.14),
                                    borderRadius: BorderRadius.circular(8)),
                                child: Text(status.toString().toUpperCase(),
                                    style: TextStyle(
                                        color: color, fontSize: 10, fontWeight: FontWeight.w700)),
                              ),
                              IconButton(
                                icon: Icon(Icons.delete_outline_rounded,
                                    color: kInkSoft.withOpacity(0.7), size: 20),
                                onPressed: () => _confirmDelete(id, word),
                              ),
                            ]),
                          ),
                        );
                      },
                    ),
        ),
      ),
    );
  }
}