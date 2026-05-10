import 'dart:convert';
import 'dart:math';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:video_player/video_player.dart';

void main() {
  runApp(const MyApp());
}

const String apiBase = "http://10.0.2.2:8000";

class QuizItem {
  final String signId;
  final String signName;
  final String category;
  final int difficulty;
  final String videoPath;
  final List<String> options;

  QuizItem({
    required this.signId,
    required this.signName,
    required this.category,
    required this.difficulty,
    required this.videoPath,
    required this.options,
  });
}

class AttemptRecord {
  final String signName;
  final int level;
  final bool correct;
  final double forgettingRisk;
  final double weakRisk;
  final double priorityScore;
  final int xp;
  final String feedback;

  AttemptRecord({
    required this.signName,
    required this.level,
    required this.correct,
    required this.forgettingRisk,
    required this.weakRisk,
    required this.priorityScore,
    required this.xp,
    required this.feedback,
  });
}

final Map<int, List<QuizItem>> levelQuizzes = {
  1: [
    QuizItem(
      signId: "S101",
      signName: "bag",
      category: "level_1_basic_objects",
      difficulty: 1,
      videoPath: "assets/videos/bag.mp4",
      options: ["bag", "book", "clock"],
    ),
    QuizItem(
      signId: "S102",
      signName: "book",
      category: "level_1_basic_objects",
      difficulty: 1,
      videoPath: "assets/videos/book.mp4",
      options: ["book", "bag", "chair"],
    ),
  ],
  2: [
    QuizItem(
      signId: "S201",
      signName: "clock",
      category: "level_2_classroom_objects",
      difficulty: 2,
      videoPath: "assets/videos/clock.mp4",
      options: ["clock", "chair", "chalk"],
    ),
    QuizItem(
      signId: "S202",
      signName: "chair",
      category: "level_2_classroom_objects",
      difficulty: 2,
      videoPath: "assets/videos/chair.mp4",
      options: ["chair", "clock", "certificate"],
    ),
  ],
  3: [
    QuizItem(
      signId: "S301",
      signName: "certificate",
      category: "level_3_advanced_school",
      difficulty: 3,
      videoPath: "assets/videos/certificate.mp4",
      options: ["certificate", "chalk", "book"],
    ),
    QuizItem(
      signId: "S302",
      signName: "chalk",
      category: "level_3_advanced_school",
      difficulty: 3,
      videoPath: "assets/videos/chalk.mp4",
      options: ["chalk", "certificate", "bag"],
    ),
  ],
};

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Adaptive Sign Quiz",
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const AdaptiveQuizScreen(),
    );
  }
}

class AdaptiveQuizScreen extends StatefulWidget {
  const AdaptiveQuizScreen({super.key});

  @override
  State<AdaptiveQuizScreen> createState() => _AdaptiveQuizScreenState();
}

class _AdaptiveQuizScreenState extends State<AdaptiveQuizScreen> {
  String selectedStudent = "STU_015";

  int currentLevel = 1;
  int currentIndex = 0;

  bool analyzing = false;
  bool quizCompleted = false;

  String? selectedAnswer;
  Map<String, dynamic>? lastResult;

  DateTime quizStartTime = DateTime.now();

  VideoPlayerController? videoController;

  final List<AttemptRecord> attempts = [];

  QuizItem get currentQuiz => levelQuizzes[currentLevel]![currentIndex];

  @override
  void initState() {
    super.initState();
    loadVideo();
  }

  @override
  void dispose() {
    videoController?.dispose();
    super.dispose();
  }

  Future<void> loadVideo() async {
    await videoController?.dispose();

    videoController = VideoPlayerController.asset(currentQuiz.videoPath);

    await videoController!.initialize();

    videoController!
      ..setLooping(true)
      ..play();

    quizStartTime = DateTime.now();

    setState(() {});
  }

  double calculateMastery(bool isCorrect, double responseTime) {
    double score = 0;

    if (isCorrect) score += 0.65;

    if (responseTime <= 4) {
      score += 0.25;
    } else if (responseTime <= 8) {
      score += 0.15;
    } else {
      score += 0.05;
    }

    score += currentLevel == 1 ? 0.10 : currentLevel == 2 ? 0.05 : 0.02;

    return score.clamp(0.0, 1.0);
  }

  double calculateConfidence(bool isCorrect, double responseTime) {
    if (!isCorrect) {
      return responseTime <= 5 ? 0.45 : 0.30;
    }

    if (responseTime <= 4) return 0.92;
    if (responseTime <= 8) return 0.78;
    return 0.62;
  }

  Future<void> submitAnswer(String answer) async {
    if (analyzing) return;

    final isCorrect = answer == currentQuiz.signName;
    final responseTime =
        DateTime.now().difference(quizStartTime).inMilliseconds / 1000;

    final masteryScore = calculateMastery(isCorrect, responseTime);
    final confidence = calculateConfidence(isCorrect, responseTime);

    setState(() {
      selectedAnswer = answer;
      analyzing = true;
      lastResult = null;
    });

    final payload = {
      "student_id": selectedStudent,
      "sign_id": currentQuiz.signId,
      "sign_name": currentQuiz.signName,
      "category": currentQuiz.category,
      "difficulty": currentQuiz.difficulty,
      "quiz_mode": "sign_to_word",
      "correct": isCorrect ? 1 : 0,
      "response_time": responseTime,
      "recognition_confidence": confidence,
      "hint_used": isCorrect ? 0 : 1,
      "attempt_number": attempts.length + 1,
      "days_since_last_review": isCorrect ? 2 : 20,
      "mastery_score": masteryScore,
    };

    try {
      final response = await http.post(
        Uri.parse("$apiBase/quiz/submit"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        attempts.add(
          AttemptRecord(
            signName: currentQuiz.signName,
            level: currentLevel,
            correct: isCorrect,
            forgettingRisk:
                (data["forgetting_probability"] as num).toDouble(),
            weakRisk: (data["weak_probability"] as num).toDouble(),
            priorityScore:
                (data["adaptive_priority_score"] as num).toDouble(),
            xp: data["gamification"]["xp_earned"],
            feedback: data["feedback"],
          ),
        );

        setState(() {
          lastResult = data;
        });
      } else {
        showMessage("Backend error: ${response.statusCode}");
      }
    } catch (e) {
      showMessage("Cannot connect to backend. Start FastAPI first.");
    } finally {
      setState(() {
        analyzing = false;
      });
    }
  }

  void nextActivity() {
    final currentLevelAttempts =
        attempts.where((a) => a.level == currentLevel).toList();

    final passedCurrentLevel =
        currentLevelAttempts.length >= 2 &&
        currentLevelAttempts.where((a) => a.correct).length == 2;

    if (currentIndex < levelQuizzes[currentLevel]!.length - 1) {
      setState(() {
        currentIndex++;
        selectedAnswer = null;
        lastResult = null;
      });
      loadVideo();
      return;
    }

    if (passedCurrentLevel && currentLevel < 3) {
      setState(() {
        currentLevel++;
        currentIndex = 0;
        selectedAnswer = null;
        lastResult = null;
      });
      loadVideo();
      return;
    }

    setState(() {
      quizCompleted = true;
    });
  }

  void restartQuiz() {
    setState(() {
      currentLevel = 1;
      currentIndex = 0;
      selectedAnswer = null;
      lastResult = null;
      quizCompleted = false;
      attempts.clear();
    });

    loadVideo();
  }

  int get totalXp => attempts.fold(0, (sum, a) => sum + a.xp);

  int get correctCount => attempts.where((a) => a.correct).length;

  double get accuracy =>
      attempts.isEmpty ? 0 : correctCount / attempts.length;

  void showMessage(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  Color riskColor(double value) {
    if (value >= 0.75) return Colors.redAccent;
    if (value >= 0.45) return Colors.orangeAccent;
    return Colors.greenAccent;
  }

  Widget sectionCard(Widget child) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 18),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xff1e293b),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xff334155)),
      ),
      child: child,
    );
  }

  Widget statCard(String title, String value, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xff0f172a),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: color.withOpacity(0.35)),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: color),
          const SizedBox(height: 8),
          Text(
            title,
            textAlign: TextAlign.center,
            style: GoogleFonts.poppins(fontSize: 12, color: Colors.white70),
          ),
          const SizedBox(height: 6),
          Text(
            value,
            textAlign: TextAlign.center,
            style: GoogleFonts.poppins(
              fontSize: 18,
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget buildVideoArea() {
    if (videoController == null || !videoController!.value.isInitialized) {
      return Container(
        height: 220,
        alignment: Alignment.center,
        child: const CircularProgressIndicator(),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(22),
      child: AspectRatio(
        aspectRatio: videoController!.value.aspectRatio,
        child: Stack(
          alignment: Alignment.bottomCenter,
          children: [
            VideoPlayer(videoController!),
            Container(
              color: Colors.black54,
              padding: const EdgeInsets.all(10),
              child: Row(
                children: [
                  IconButton(
                    onPressed: () {
                      setState(() {
                        videoController!.value.isPlaying
                            ? videoController!.pause()
                            : videoController!.play();
                      });
                    },
                    icon: Icon(
                      videoController!.value.isPlaying
                          ? Icons.pause_circle
                          : Icons.play_circle,
                      color: Colors.cyanAccent,
                    ),
                  ),
                  Text(
                    "Watch sign and select correct answer",
                    style: GoogleFonts.poppins(fontSize: 12),
                  ),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }

  Widget buildQuizScreen() {
    final shuffledOptions = [...currentQuiz.options]..shuffle(Random(3));

    return SingleChildScrollView(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          sectionCard(
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Adaptive Sign Quiz",
                  style: GoogleFonts.poppins(
                    fontSize: 26,
                    fontWeight: FontWeight.bold,
                    color: Colors.cyanAccent,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  "Level $currentLevel • Question ${currentIndex + 1}/2",
                  style: GoogleFonts.poppins(color: Colors.white70),
                ),
                const SizedBox(height: 16),
                LinearProgressIndicator(
                  value: ((currentLevel - 1) * 2 + currentIndex + 1) / 6,
                  color: Colors.cyanAccent,
                  backgroundColor: Colors.white12,
                ),
              ],
            ),
          ),

          sectionCard(
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Sign Video",
                  style: GoogleFonts.poppins(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 14),
                buildVideoArea(),
                const SizedBox(height: 14),
                Text(
                  "What is the correct answer?",
                  style: GoogleFonts.poppins(color: Colors.white70),
                ),
              ],
            ),
          ),

          sectionCard(
            Column(
              children: shuffledOptions.map((option) {
                final isSelected = selectedAnswer == option;
                final isCorrectOption = option == currentQuiz.signName;

                Color color = const Color(0xff0f172a);

                if (selectedAnswer != null) {
                  if (isCorrectOption) {
                    color = Colors.green.withOpacity(0.35);
                  } else if (isSelected) {
                    color = Colors.red.withOpacity(0.35);
                  }
                }

                return Container(
                  width: double.infinity,
                  margin: const EdgeInsets.only(bottom: 12),
                  child: ElevatedButton(
                    onPressed: selectedAnswer == null
                        ? () => submitAnswer(option)
                        : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: color,
                      disabledBackgroundColor: color,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(18),
                        side: const BorderSide(color: Color(0xff334155)),
                      ),
                    ),
                    child: Text(
                      option.toUpperCase(),
                      style: GoogleFonts.poppins(
                        fontSize: 17,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
          ),

          if (analyzing)
            sectionCard(
              const Center(
                child: CircularProgressIndicator(color: Colors.cyanAccent),
              ),
            ),

          if (lastResult != null) buildPredictionOutput(),
        ],
      ),
    );
  }

  Widget buildPredictionOutput() {
    final forgettingRisk =
        (lastResult!["forgetting_probability"] as num).toDouble();
    final weakRisk = (lastResult!["weak_probability"] as num).toDouble();
    final priority =
        (lastResult!["adaptive_priority_score"] as num).toDouble();

    return sectionCard(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "ML Prediction Output",
            style: GoogleFonts.poppins(
              fontSize: 20,
              color: Colors.cyanAccent,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          GridView.count(
            crossAxisCount: 2,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            crossAxisSpacing: 12,
            mainAxisSpacing: 12,
            children: [
              statCard(
                "Result",
                lastResult!["result"].toString().toUpperCase(),
                Icons.fact_check,
                lastResult!["result"] == "correct"
                    ? Colors.greenAccent
                    : Colors.redAccent,
              ),
              statCard(
                "Forgetting Risk",
                "${(forgettingRisk * 100).toStringAsFixed(1)}%",
                Icons.psychology,
                riskColor(forgettingRisk),
              ),
              statCard(
                "Weak Sign Risk",
                "${(weakRisk * 100).toStringAsFixed(1)}%",
                Icons.warning,
                riskColor(weakRisk),
              ),
              statCard(
                "Priority",
                "${(priority * 100).toStringAsFixed(1)}%",
                Icons.priority_high,
                riskColor(priority),
              ),
            ],
          ),
          const SizedBox(height: 18),
          Text(
            "AI Feedback",
            style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            lastResult!["feedback"],
            style: GoogleFonts.poppins(color: Colors.white70),
          ),
          const SizedBox(height: 18),
          SizedBox(
            width: double.infinity,
            height: 54,
            child: ElevatedButton.icon(
              onPressed: nextActivity,
              icon: const Icon(Icons.arrow_forward),
              label: Text(
                "Continue Next Activity",
                style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.cyanAccent,
                foregroundColor: Colors.black,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget buildAnalysisScreen() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          sectionCard(
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Student Learning Analysis",
                  style: GoogleFonts.poppins(
                    fontSize: 25,
                    fontWeight: FontWeight.bold,
                    color: Colors.cyanAccent,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  "Final adaptive quiz report based on ML outputs.",
                  style: GoogleFonts.poppins(color: Colors.white70),
                ),
              ],
            ),
          ),

          sectionCard(
            GridView.count(
              crossAxisCount: 2,
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
              children: [
                statCard(
                  "Accuracy",
                  "${(accuracy * 100).toStringAsFixed(1)}%",
                  Icons.check_circle,
                  Colors.greenAccent,
                ),
                statCard(
                  "Total XP",
                  "$totalXp",
                  Icons.stars,
                  Colors.amberAccent,
                ),
                statCard(
                  "Correct",
                  "$correctCount/${attempts.length}",
                  Icons.task_alt,
                  Colors.cyanAccent,
                ),
                statCard(
                  "Level Reached",
                  "$currentLevel",
                  Icons.trending_up,
                  Colors.purpleAccent,
                ),
              ],
            ),
          ),

          sectionCard(
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Risk Analysis Chart",
                  style: GoogleFonts.poppins(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 18),
                SizedBox(
                  height: 260,
                  child: BarChart(
                    BarChartData(
                      maxY: 1,
                      barGroups: attempts.asMap().entries.map((entry) {
                        final i = entry.key;
                        final a = entry.value;

                        return BarChartGroupData(
                          x: i,
                          barRods: [
                            BarChartRodData(
                              toY: a.forgettingRisk,
                              width: 8,
                              color: Colors.orangeAccent,
                            ),
                            BarChartRodData(
                              toY: a.weakRisk,
                              width: 8,
                              color: Colors.redAccent,
                            ),
                            BarChartRodData(
                              toY: a.priorityScore,
                              width: 8,
                              color: Colors.cyanAccent,
                            ),
                          ],
                        );
                      }).toList(),
                      titlesData: const FlTitlesData(
                        topTitles: AxisTitles(
                          sideTitles: SideTitles(showTitles: false),
                        ),
                        rightTitles: AxisTitles(
                          sideTitles: SideTitles(showTitles: false),
                        ),
                      ),
                      borderData: FlBorderData(show: false),
                      gridData: const FlGridData(show: true),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  "Orange = Forgetting Risk, Red = Weak Sign Risk, Cyan = Adaptive Priority",
                  style: GoogleFonts.poppins(
                    color: Colors.white70,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),

          sectionCard(
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Detailed Attempt Report",
                  style: GoogleFonts.poppins(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 12),
                ...attempts.map((a) {
                  return Container(
                    margin: const EdgeInsets.only(bottom: 12),
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: const Color(0xff0f172a),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Text(
                      "Level ${a.level} • ${a.signName.toUpperCase()}\n"
                      "Result: ${a.correct ? "Correct" : "Incorrect"} | XP: ${a.xp}\n"
                      "Forgetting: ${(a.forgettingRisk * 100).toStringAsFixed(1)}% | "
                      "Weak: ${(a.weakRisk * 100).toStringAsFixed(1)}% | "
                      "Priority: ${(a.priorityScore * 100).toStringAsFixed(1)}%",
                      style: GoogleFonts.poppins(height: 1.5),
                    ),
                  );
                }),
              ],
            ),
          ),

          SizedBox(
            width: double.infinity,
            height: 54,
            child: ElevatedButton.icon(
              onPressed: restartQuiz,
              icon: const Icon(Icons.restart_alt),
              label: Text(
                "Restart Activity",
                style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.cyanAccent,
                foregroundColor: Colors.black,
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xff020617),
      appBar: AppBar(
        backgroundColor: const Color(0xff0f172a),
        title: Text(
          quizCompleted ? "Learning Report" : "Adaptive Quiz Activity",
          style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
        ),
      ),
      body: quizCompleted ? buildAnalysisScreen() : buildQuizScreen(),
    );
  }
}