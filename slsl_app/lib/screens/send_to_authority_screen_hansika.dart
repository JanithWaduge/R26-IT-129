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
  List<Map<String, dynamic>> _signs = [];
  final Set<String> _selectedIds = {}; // ── NEW: tracks checked signs ──
  int _totalAwaitingDecision = 0;
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
    final signs = List<Map<String, dynamic>>.from(data['signs'] ?? []);
    setState(() {
      _signs = signs;
      _totalAwaitingDecision = data['total_awaiting_decision'] ?? 0;
      _selectedIds.clear();
      _loading = false;
    });
  }

  bool _isValidEmail(String email) =>
      email.contains('@') && email.contains('.') && email.trim().length > 5;

  void _toggleSelect(String id) {
    setState(() {
      if (_selectedIds.contains(id)) {
        _selectedIds.remove(id);
      } else {
        _selectedIds.add(id);
      }
    });
  }

  void _selectAll() {
    setState(() {
      if (_selectedIds.length == _signs.length) {
        _selectedIds.clear();
      } else {
        _selectedIds
          ..clear()
          ..addAll(_signs.map((s) => s['_id'].toString()));
      }
    });
  }

  Future<void> _send() async {
    final email = _emailController.text.trim();
    if (!_isValidEmail(email)) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text('Please enter a valid email address.'),
        backgroundColor: kError,
      ));
      return;
    }
    if (_selectedIds.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text('Select at least one sign to send.'),
        backgroundColor: kError,
      ));
      return;
    }

    setState(() => _sending = true);
    final result = await TeacherApiService.sendToAuthority(
      authorityEmail: email,
      submissionIds: _selectedIds.toList(),
    );
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
    final allSelected = _signs.isNotEmpty && _selectedIds.length == _signs.length;

    return Column(
      children: [
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              Container(
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: (_signs.isEmpty ? kWarning : kPrimary).withOpacity(0.08),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: (_signs.isEmpty ? kWarning : kPrimary).withOpacity(0.25)),
                ),
                child: Row(children: [
                  Container(
                    width: 46, height: 46,
                    decoration: BoxDecoration(
                        color: (_signs.isEmpty ? kWarning : kPrimary).withOpacity(0.16),
                        shape: BoxShape.circle),
                    child: Icon(
                        _signs.isEmpty ? Icons.hourglass_top_rounded : Icons.mark_email_unread_rounded,
                        color: _signs.isEmpty ? kWarning : kPrimary, size: 24),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _signs.isEmpty ? 'No signs waiting to be sent' : '${_signs.length} sign(s) available to send',
                          style: TextStyle(
                              color: _signs.isEmpty ? kWarning : kPrimary,
                              fontSize: 15,
                              fontWeight: FontWeight.w700),
                        ),
                        const SizedBox(height: 4),
                        Text('${_selectedIds.length} selected',
                            style: const TextStyle(color: kInkSoft, fontSize: 12)),
                        if (_totalAwaitingDecision > _signs.length) ...[
                          const SizedBox(height: 6),
                          Text(
                            '${_totalAwaitingDecision - _signs.length} sign(s) already sent, still awaiting the authority\'s decision.',
                            style: const TextStyle(color: kInkSoft, fontSize: 11),
                          ),
                        ],
                      ],
                    ),
                  ),
                ]),
              ),
              const SizedBox(height: 24),
              if (_signs.isNotEmpty)
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('Select signs to send',
                        style: TextStyle(
                            color: kInkSoft, fontSize: 11, fontWeight: FontWeight.w700, letterSpacing: 1.5)),
                    TextButton(
                      onPressed: _selectAll,
                      child: Text(allSelected ? 'Deselect All' : 'Select All',
                          style: const TextStyle(color: kPrimary, fontSize: 12, fontWeight: FontWeight.w600)),
                    ),
                  ],
                ),
              const SizedBox(height: 8),
              if (_signs.isEmpty)
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(color: kSurface, borderRadius: BorderRadius.circular(14)),
                  child: const Text(
                    'Record a new sign first — it will appear here once submitted.',
                    style: TextStyle(color: kInkSoft, fontSize: 12),
                  ),
                )
              else
                ..._signs.map((s) {
                  final id = s['_id'].toString();
                  final selected = _selectedIds.contains(id);
                  return GestureDetector(
                    onTap: () => _toggleSelect(id),
                    child: Container(
                      margin: const EdgeInsets.only(bottom: 8),
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                      decoration: BoxDecoration(
                        color: selected ? kPrimary.withOpacity(0.08) : Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: selected ? kPrimary : const Color(0xFFEDEFF5)),
                        boxShadow: [
                          BoxShadow(color: kPrimary.withOpacity(0.05), blurRadius: 8, offset: const Offset(0, 3)),
                        ],
                      ),
                      child: Row(children: [
                        Checkbox(
                          value: selected,
                          activeColor: kPrimary,
                          onChanged: (_) => _toggleSelect(id),
                        ),
                        Icon(Icons.gesture_rounded, color: kPrimary, size: 18),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(s['english_word'] ?? '',
                                  style: const TextStyle(color: kInk, fontSize: 13, fontWeight: FontWeight.w600)),
                              if (s['sinhala_word'] != null)
                                Text(s['sinhala_word'],
                                    style: const TextStyle(color: kInkSoft, fontSize: 11)),
                            ],
                          ),
                        ),
                        Text((s['category'] ?? '').toString().toUpperCase(),
                            style: TextStyle(
                                color: kPrimary.withOpacity(0.75), fontSize: 10, fontWeight: FontWeight.w700)),
                      ]),
                    ),
                  );
                }),
              const SizedBox(height: 24),
              const Text('Authority email',
                  style: TextStyle(color: kInkSoft, fontSize: 11, fontWeight: FontWeight.w700, letterSpacing: 1.5)),
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
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                ),
              ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          child: SizedBox(
            width: double.infinity,
            height: 54,
            child: DecoratedBox(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(14),
                gradient: (_selectedIds.isEmpty || _sending)
                    ? null
                    : const LinearGradient(colors: [kPrimary, kSecondary]),
                color: (_selectedIds.isEmpty || _sending) ? const Color(0xFFEDEFF5) : null,
              ),
              child: ElevatedButton(
                onPressed: (_selectedIds.isEmpty || _sending) ? null : _send,
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
                        width: 22, height: 22,
                        child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                    : Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.send_rounded),
                          const SizedBox(width: 10),
                          Text(
                            _selectedIds.isEmpty
                                ? 'Send to Authority'
                                : 'Send ${_selectedIds.length} sign${_selectedIds.length == 1 ? '' : 's'} to Authority',
                            style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
                          ),
                        ],
                      ),
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