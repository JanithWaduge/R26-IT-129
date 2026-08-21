import 'package:flutter/material.dart';

import '../../../core/theme/app_colors.dart';
import 'auth_controller.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({required this.controller, super.key});

  final AuthController controller;

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();

  final TextEditingController _nameController = TextEditingController();

  final TextEditingController _emailController = TextEditingController();

  final TextEditingController _passwordController = TextEditingController();

  final TextEditingController _gradeController = TextEditingController();

  String _language = 'sinhala';
  bool _obscurePassword = true;

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _gradeController.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    final bool success = await widget.controller.register(
      fullName: _nameController.text,
      email: _emailController.text,
      password: _passwordController.text,
      preferredLanguage: _language,
      gradeLevel: _gradeController.text,
    );

    if (!mounted) {
      return;
    }

    if (success) {
      Navigator.of(context).pop();
      return;
    }

    final String? error = widget.controller.errorMessage;

    if (error != null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(error), backgroundColor: AppColors.error));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(title: const Text('Create Account')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  TextFormField(
                    controller: _nameController,
                    textCapitalization: TextCapitalization.words,
                    style: const TextStyle(color: AppColors.ink),
                    decoration: const InputDecoration(
                      labelText: 'Full name',
                      prefixIcon: Icon(Icons.person, color: AppColors.secondary),
                    ),
                    validator: (String? value) {
                      if ((value?.trim().length ?? 0) < 2) {
                        return 'Enter the '
                            'student name.';
                      }

                      return null;
                    },
                  ),
                  const SizedBox(height: 16),
                  TextFormField(
                    controller: _emailController,
                    keyboardType: TextInputType.emailAddress,
                    style: const TextStyle(color: AppColors.ink),
                    decoration: const InputDecoration(
                      labelText: 'Email address',
                      prefixIcon: Icon(Icons.email, color: AppColors.primary),
                    ),
                    validator: (String? value) {
                      final String email = value?.trim() ?? '';

                      if (email.isEmpty || !email.contains('@')) {
                        return 'Enter a valid '
                            'email address.';
                      }

                      return null;
                    },
                  ),
                  const SizedBox(height: 16),
                  TextFormField(
                    controller: _passwordController,
                    obscureText: _obscurePassword,
                    style: const TextStyle(color: AppColors.ink),
                    decoration: InputDecoration(
                      labelText: 'Password',
                      helperText:
                          'At least 10 characters '
                          'with a letter and number.',
                      prefixIcon: const Icon(Icons.lock, color: AppColors.primary),
                      suffixIcon: IconButton(
                        onPressed: () {
                          setState(() {
                            _obscurePassword = !_obscurePassword;
                          });
                        },
                        icon: Icon(
                          _obscurePassword
                              ? Icons.visibility
                              : Icons.visibility_off,
                          color: AppColors.inkSoft,
                        ),
                      ),
                    ),
                    validator: (String? value) {
                      final String password = value ?? '';

                      if (password.length < 10) {
                        return 'Use at least '
                            '10 characters.';
                      }

                      if (!RegExp(r'[A-Za-z]').hasMatch(password)) {
                        return 'Add at least '
                            'one letter.';
                      }

                      if (!RegExp(r'[0-9]').hasMatch(password)) {
                        return 'Add at least '
                            'one number.';
                      }

                      return null;
                    },
                  ),
                  const SizedBox(height: 16),
                  DropdownButtonFormField<String>(
                    value: _language,
                    style: const TextStyle(color: AppColors.ink, fontSize: 15),
                    decoration: const InputDecoration(
                      labelText: 'Preferred language',
                      prefixIcon: Icon(Icons.language, color: AppColors.success),
                    ),
                    items: const [
                      DropdownMenuItem(
                        value: 'sinhala',
                        child: Text('Sinhala'),
                      ),
                      DropdownMenuItem(value: 'tamil', child: Text('Tamil')),
                      DropdownMenuItem(
                        value: 'english',
                        child: Text('English'),
                      ),
                    ],
                    onChanged: (String? value) {
                      if (value != null) {
                        setState(() {
                          _language = value;
                        });
                      }
                    },
                  ),
                  const SizedBox(height: 16),
                  TextFormField(
                    controller: _gradeController,
                    style: const TextStyle(color: AppColors.ink),
                    decoration: const InputDecoration(
                      labelText: 'Grade or level',
                      hintText: 'Grade 8',
                      prefixIcon: Icon(Icons.school, color: AppColors.indigo),
                    ),
                    validator: (String? value) {
                      if ((value?.trim().isEmpty ?? true)) {
                        return 'Enter the '
                            'student level.';
                      }

                      return null;
                    },
                  ),
                  const SizedBox(height: 24),
                  ListenableBuilder(
                    listenable: widget.controller,
                    builder: (context, child) {
                      return SizedBox(
                        height: 52,
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(14),
                            gradient: widget.controller.isSubmitting
                                ? null
                                : AppColors.primaryGradient,
                            color: widget.controller.isSubmitting
                                ? AppColors.hairline
                                : null,
                          ),
                          child: FilledButton(
                            onPressed: widget.controller.isSubmitting
                                ? null
                                : _submit,
                            style: FilledButton.styleFrom(
                              backgroundColor: Colors.transparent,
                              shadowColor: Colors.transparent,
                              disabledBackgroundColor: Colors.transparent,
                            ),
                            child: widget.controller.isSubmitting
                                ? const SizedBox(
                                    width: 22,
                                    height: 22,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: Colors.white,
                                    ),
                                  )
                                : const Text('Create Account'),
                          ),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}