// lib/screens/send_to_authority_screen_hansika.dart
import 'package:flutter/material.dart';
import '../constants.dart';
import '../services/teacher_api_service.dart';

class SendToAuthorityScreenHansika extends StatefulWidget {
  const SendToAuthorityScreenHansika({super.key});
  @override
  State<SendToAuthorityScreenHansika> createState() =>
      _SendToAuthorityScreenHansikaState();
}

class _SendToAuthorityScreenHansikaState
    extends State<SendToAuthorityScreenHansika> {
  final _emailController = TextEditingController();
  bool _loading = true;
  bool _sending = false;
  int _count = 0;
  bool _ready = false;
  int _totalAwaitingDecision = 0;
  List<Map<String, dynamic>> _signs = [];
  Map<String, dynamic>? _result;

  @override
  void initState() {
    super.initState();
    _loadBatch();
  }

  @override
  void dispose() {
    _emailController.dispose();
    super.dispose();
  }

  Future<void> _loadBatch() async {
    setState(() => _loading = true);
    final data = await TeacherApiService.getPendingBatch();
    if (!mounted) return;
    setState(() {
      _count = data['count'] ?? 0;
      _ready = data['ready'] == true;
      _signs = List<Map<String, dynamic>>.from(data['signs'] ?? []);
      _totalAwaitingDecision = data['total_awaiting_decision'] ?? 0;
      _loading = false;
    });
  }

  bool _isValidEmail(String email) =>
      email.contains('@') && email.contains('.') && email.trim().length > 5;

  Future<void> _send() async {
    final email = _emailController.text.trim();
    if (!_isValidEmail(email)) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text('Please enter a valid email address.'),
        backgroundColor: kError,
      ));
      return;
    }

    setState(() => _sending = true);
    final result = await TeacherApiService.sendToAuthority(email);
    if (!mounted) return;
    setState(() {
      _sending = false;
      _result = result;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBackground,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text('Send to Authority', style: TextStyle(color: kInk, fontWeight: FontWeight.w700)),
        iconTheme: const IconThemeData(color: kInk),
      ),
      body: SafeArea(
        child: _loading
            ? const Center(child: CircularProgressIndicator(color: kPrimary))
            : _result != null
                ? _buildResultView()
                : _buildFormView(),
      ),
    );
  }

  Widget _buildFormView() {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: (_ready ? kSuccess : kWarning).withOpacity(0.08),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: (_ready ? kSuccess : kWarning).withOpacity(0.25)),
          ),
          child: Row(children: [
            Container(
              width: 46, height: 46,
              decoration: BoxDecoration(
                  color: (_ready ? kSuccess : kWarning).withOpacity(0.16), shape: BoxShape.circle),
              child: Icon(_ready ? Icons.mark_email_read_rounded : Icons.hourglass_top_rounded,
                  color: _ready ? kSuccess : kWarning, size: 24),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _ready ? 'Batch ready to send!' : 'Waiting for more signs',
                    style: TextStyle(
                        color: _ready ? kSuccess : kWarning,
                        fontSize: 15,
                        fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 4),
                  Text('$_count new sign${_count == 1 ? '' : 's'} not yet emailed',
                      style: const TextStyle(color: kInkSoft, fontSize: 12)),
                  // ── Clarifies the "0 pending" vs dashboard count confusion ──
                  if (_count == 0 && _totalAwaitingDecision > 0) ...[
                    const SizedBox(height: 6),
                    Text(
                      '$_totalAwaitingDecision sign${_totalAwaitingDecision == 1 ? '' : 's'} already sent, still awaiting the authority\'s decision.',
                      style: const TextStyle(color: kInkSoft, fontSize: 11),
                    ),
                  ],
                ],
              ),
            ),
          ]),
        ),
        const SizedBox(height: 24),
        const Text('Signs in this batch',
            style: TextStyle(
                color: kInkSoft,
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.5)),
        const SizedBox(height: 12),
        if (_signs.isEmpty)
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: kSurface, borderRadius: BorderRadius.circular(14)),
            child: Text(
              _totalAwaitingDecision > 0
                  ? 'No new signs waiting to be sent — record more to start a new batch.'
                  : 'No pending signs right now.',
              style: const TextStyle(color: kInkSoft, fontSize: 12),
            ),
          )
        else
          ..._signs.map((s) => Container(
                margin: const EdgeInsets.only(bottom: 8),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: kPrimary.withOpacity(0.18)),
                  boxShadow: [
                    BoxShadow(color: kPrimary.withOpacity(0.06), blurRadius: 10, offset: const Offset(0, 4)),
                  ],
                ),
                child: Row(children: [
                  Icon(Icons.gesture_rounded, color: kPrimary, size: 18),
                  const SizedBox(width: 10),
                  Expanded(
                      child: Text(s['english_word'] ?? '',
                          style: const TextStyle(
                              color: kInk, fontSize: 13, fontWeight: FontWeight.w600))),
                  Text((s['category'] ?? '').toString().toUpperCase(),
                      style: TextStyle(
                          color: kPrimary.withOpacity(0.75),
                          fontSize: 10,
                          fontWeight: FontWeight.w700)),
                ]),
              )),
        const SizedBox(height: 28),
        const Text('Authority email',
            style: TextStyle(
                color: kInkSoft,
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.5)),
        const SizedBox(height: 10),
        TextField(
          controller: _emailController,
          keyboardType: TextInputType.emailAddress,
          style: const TextStyle(color: kInk),
          decoration: InputDecoration(
            hintText: 'authority@school.lk',
            hintStyle: const TextStyle(color: kInkSoft),
            prefixIcon: const Icon(Icons.email_outlined, color: kPrimary),
            filled: true,
            fillColor: kSurface,
            border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
          ),
        ),
        const SizedBox(height: 24),
        SizedBox(
          width: double.infinity,
          height: 54,
          child: DecoratedBox(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(14),
              gradient: (_signs.isEmpty || _sending)
                  ? null
                  : const LinearGradient(colors: [kPrimary, kSecondary]),
              color: (_signs.isEmpty || _sending) ? const Color(0xFFEDEFF5) : null,
            ),
            child: ElevatedButton(
              onPressed: (_signs.isEmpty || _sending) ? null : _send,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.transparent,
                shadowColor: Colors.transparent,
                disabledBackgroundColor: Colors.transparent,
                foregroundColor: Colors.white,
                disabledForegroundColor: kInkSoft,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
              child: _sending
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                  : const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.send_rounded),
                        SizedBox(width: 10),
                        Text('Send to Authority',
                            style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
                      ],
                    ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildResultView() {
    final sent = _result!['sent'] == true;
    final message = _result!['message'] ?? _result!['error'] ?? 'Something went wrong.';
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 96, height: 96,
            decoration: BoxDecoration(
              color: (sent ? kSuccess : kWarning).withOpacity(0.12),
              shape: BoxShape.circle,
            ),
            child: Icon(
              sent ? Icons.check_circle_rounded : Icons.error_outline_rounded,
              color: sent ? kSuccess : kWarning,
              size: 52,
            ),
          ),
          const SizedBox(height: 20),
          Text(
            sent ? 'Sent to Authority!' : 'Something Went Wrong',
            style: const TextStyle(color: kInk, fontSize: 20, fontWeight: FontWeight.w700),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 10),
          Text(message.toString(),
              style: const TextStyle(color: kInkSoft, fontSize: 13),
              textAlign: TextAlign.center),
          const SizedBox(height: 32),
          SizedBox(
            width: double.infinity,
            height: 52,
            child: DecoratedBox(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(14),
                gradient: const LinearGradient(colors: [kPrimary, kSecondary]),
              ),
              child: ElevatedButton(
                onPressed: () => Navigator.pop(context),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.transparent,
                  shadowColor: Colors.transparent,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                ),
                child: const Text('Back to Dashboard', style: TextStyle(fontWeight: FontWeight.w700)),
              ),
            ),
          ),
        ],
      ),
    );
  }
}