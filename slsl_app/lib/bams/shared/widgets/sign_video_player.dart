import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../../core/config/api_config.dart';
import '../../core/theme/app_colors.dart';

class SignVideoPlayer extends StatefulWidget {
  const SignVideoPlayer({
    required this.mediaUri,
    this.autoPlay = true,
    this.loop = true,
    super.key,
  });

  final String mediaUri;
  final bool autoPlay;
  final bool loop;

  @override
  State<SignVideoPlayer> createState() => _SignVideoPlayerState();
}

class _SignVideoPlayerState extends State<SignVideoPlayer> {
  VideoPlayerController? _controller;

  Future<void>? _initialization;
  String? _error;

  @override
  void initState() {
    super.initState();

    _initialize();
  }

  @override
  void didUpdateWidget(SignVideoPlayer oldWidget) {
    super.didUpdateWidget(oldWidget);

    if (oldWidget.mediaUri != widget.mediaUri) {
      _disposeController();
      _initialize();
    }
  }

  void _initialize() {
    final String resolvedUrl = ApiConfig.resolveMediaUrl(widget.mediaUri);

    final VideoPlayerController controller = VideoPlayerController.networkUrl(
      Uri.parse(resolvedUrl),
    );

    _controller = controller;

    _initialization = controller
        .initialize()
        .then((_) async {
          await controller.setLooping(widget.loop);

          if (widget.autoPlay) {
            await controller.play();
          }

          if (mounted) {
            setState(() {});
          }
        })
        .catchError((Object error) {
          if (mounted) {
            setState(() {
              _error = 'Unable to load sign video.';
            });
          }

          throw error;
        });
  }

  void _disposeController() {
    final VideoPlayerController? controller = _controller;

    _controller = null;

    controller?.dispose();
  }

  @override
  void dispose() {
    _disposeController();
    super.dispose();
  }

  Future<void> _togglePlayback() async {
    final VideoPlayerController? controller = _controller;

    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    if (controller.value.isPlaying) {
      await controller.pause();
    } else {
      await controller.play();
    }

    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _replay() async {
    final VideoPlayerController? controller = _controller;

    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    await controller.seekTo(Duration.zero);

    await controller.play();

    if (mounted) {
      setState(() {});
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return _VideoMessage(icon: Icons.error_outline, message: _error!);
    }

    final Future<void>? initialization = _initialization;

    final VideoPlayerController? controller = _controller;

    if (initialization == null || controller == null) {
      return const _VideoMessage(
        icon: Icons.video_library_outlined,
        message: 'Video is unavailable.',
      );
    }

    return FutureBuilder<void>(
      future: initialization,
      builder: (BuildContext context, AsyncSnapshot<void> snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const SizedBox(
            height: 230,
            child: Center(child: CircularProgressIndicator()),
          );
        }

        if (snapshot.hasError) {
          return const _VideoMessage(
            icon: Icons.error_outline,
            message: 'Unable to play this video.',
          );
        }

        final double aspectRatio = controller.value.aspectRatio.isFinite
            ? controller.value.aspectRatio
            : 16 / 9;

        return Column(
          children: [
            Container(
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppColors.hairline),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: AspectRatio(
                  aspectRatio: aspectRatio == 0 ? 16 / 9 : aspectRatio,
                  child: VideoPlayer(controller),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  style: IconButton.styleFrom(
                    backgroundColor: AppColors.surface,
                    foregroundColor: AppColors.ink,
                    side: const BorderSide(color: AppColors.hairline),
                  ),
                  onPressed: _togglePlayback,
                  tooltip: controller.value.isPlaying ? 'Pause' : 'Play',
                  icon: Icon(
                    controller.value.isPlaying ? Icons.pause : Icons.play_arrow,
                  ),
                ),
                const SizedBox(width: 8),
                IconButton(
                  style: IconButton.styleFrom(
                    backgroundColor: AppColors.surface,
                    foregroundColor: AppColors.ink,
                    side: const BorderSide(color: AppColors.hairline),
                  ),
                  onPressed: _replay,
                  tooltip: 'Replay',
                  icon: const Icon(Icons.replay),
                ),
              ],
            ),
          ],
        );
      },
    );
  }
}

class _VideoMessage extends StatelessWidget {
  const _VideoMessage({required this.icon, required this.message});

  final IconData icon;
  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 230,
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.hairline),
      ),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 64, color: AppColors.inkSoft),
            const SizedBox(height: 12),
            Text(
              message,
              textAlign: TextAlign.center,
              style: const TextStyle(color: AppColors.inkSoft),
            ),
          ],
        ),
      ),
    );
  }
}
