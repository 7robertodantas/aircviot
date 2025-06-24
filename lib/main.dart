import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:logger/logger.dart';

final logger = Logger();
List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TFLite Detection (Person Only)',
      theme: ThemeData(primarySwatch: Colors.deepPurple),
      home: const DetectPage(),
    );
  }
}

class DetectPage extends StatefulWidget {
  const DetectPage({Key? key}) : super(key: key);

  @override
  State<DetectPage> createState() => _DetectPageState();
}

class _DetectPageState extends State<DetectPage> {
  CameraController? _cameraController;
  int _selectedCameraIdx = 0;
  Interpreter? _interpreter;
  int inputSize = 300;
  bool _isDetecting = false;
  List<Map<String, dynamic>> _results = [];
  Timer? _timer;

  final List<String> cocoLabels = ["person"];

  @override
  void initState() {
    super.initState();
    logger.i("App started, initializing...");
    _initEverything();
  }

  Future<void> _initEverything() async {
    logger.i("Loading TFLite model...");
    await _loadModel();
    logger.i("Initializing camera...");
    await _initCamera(_selectedCameraIdx);
    logger.i("Initialization complete. Starting periodic detection.");
    _timer = Timer.periodic(const Duration(seconds: 1), (_) => _runDetection());
  }

  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/ssd_mobilenet.tflite');
    logger.i("TFLite model loaded successfully.");
    setState(() {});
  }

  Future<void> _initCamera(int idx) async {
    logger.i("Disposing current camera controller...");
    await _cameraController?.dispose();
    logger.i("Setting up camera index $idx...");
    final controller = CameraController(
      cameras[idx],
      ResolutionPreset.medium,
      enableAudio: false,
    );
    try {
      await controller.initialize();
      logger.i("Camera index $idx initialized successfully.");
      setState(() => _cameraController = controller);
    } catch (e) {
      logger.e("Error initializing camera: $e");
    }
  }

  void _switchCamera() async {
    logger.i("Switching camera...");
    int newIdx = (_selectedCameraIdx + 1) % cameras.length;
    _timer?.cancel();
    _isDetecting = false;
    try {
      await _initCamera(newIdx);
      setState(() => _selectedCameraIdx = newIdx);
      _timer = Timer.periodic(const Duration(seconds: 1), (_) => _runDetection());
      logger.i("Switched to camera index $newIdx.");
    } catch (e) {
      logger.e("Failed to switch camera: $e");
    }
  }

  Future<void> _runDetection() async {
    logger.d("Running detection...");
    if (_interpreter == null || _cameraController == null || _isDetecting) {
      logger.w("Detection skipped: Interpreter or camera not ready.");
      return;
    }
    if (!_cameraController!.value.isInitialized) {
      logger.w("Camera not initialized.");
      return;
    }

    _isDetecting = true;
    try {
      XFile file = await _cameraController!.takePicture();
      logger.d("Image captured.");
      final bytes = await file.readAsBytes();

      final img.Image? image = img.decodeImage(bytes);
      if (image == null) {
        logger.w("Image decoding failed.");
        _isDetecting = false;
        return;
      }

      final img.Image resized = img.copyResize(image, width: inputSize, height: inputSize);
      logger.d("Image resized to $inputSize x $inputSize.");

      var input = imageToByteListUint8(resized, inputSize);
      var inputList = input.reshape([1, inputSize, inputSize, 3]);

      var outputBoxes = List.filled(1 * 10 * 4, 0.0).reshape([1, 10, 4]);
      var outputClasses = List.filled(1 * 10, 0.0).reshape([1, 10]);
      var outputScores = List.filled(1 * 10, 0.0).reshape([1, 10]);
      var numDetections = List.filled(1, 0.0).reshape([1]);

      var outputs = {
        0: outputBoxes,
        1: outputClasses,
        2: outputScores,
        3: numDetections
      };

      _interpreter!.runForMultipleInputs([inputList], outputs);

      final boxes = outputs[0] as List;
      final classes = outputs[1] as List;
      final scores = outputs[2] as List;
      final count = ((outputs[3] as List)[0] as double).toInt();
      logger.i("Model inference complete. Detections: $count");

      List<Map<String, dynamic>> detections = [];
      for (int i = 0; i < count; i++) {
        int cls = (classes[0][i] as double).toInt();
        double score = scores[0][i];
        var rect = boxes[0][i];
        logger.d('Detection $i -> class: $cls, score: ${score.toStringAsFixed(2)}, rect: $rect');

        if (cls == 0 && score > 0.6) {
          detections.add({'score': score, 'rect': rect, 'class': cls});
        }
      }
      setState(() => _results = detections);
    } catch (e) {
      logger.e('Detection error: $e');
    }
    _isDetecting = false;
  }

  @override
  void dispose() {
    logger.i("Disposing camera and interpreter...");
    _cameraController?.dispose();
    _timer?.cancel();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('TFLite Detection (Person Only)'),
        actions: [
          IconButton(
            icon: const Icon(Icons.switch_camera),
            onPressed: _switchCamera,
          ),
        ],
      ),
      body: Stack(
        children: [
          CameraPreview(_cameraController!),
          BoundingBoxOverlay(
            results: _results,
            previewSize: _cameraController!.value.previewSize!,
            screenSize: MediaQuery.of(context).size,
            cocoLabels: cocoLabels,
          ),
          Positioned(
            top: 40,
            left: 20,
            child: Container(
              padding: const EdgeInsets.all(8),
              color: Colors.black54,
              child: Text(
                'Persons detected: ${_results.length}',
                style: const TextStyle(color: Colors.white, fontSize: 20),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

Uint8List imageToByteListUint8(img.Image image, int inputSize) {
  var convertedBytes = Uint8List(inputSize * inputSize * 3);
  int pixelIndex = 0;
  for (int y = 0; y < inputSize; y++) {
    for (int x = 0; x < inputSize; x++) {
      final pixel = image.getPixel(x, y);
      convertedBytes[pixelIndex++] = pixel.r.toInt();
      convertedBytes[pixelIndex++] = pixel.g.toInt();
      convertedBytes[pixelIndex++] = pixel.b.toInt();
    }
  }
  return convertedBytes;
}

class BoundingBoxOverlay extends StatelessWidget {
  final List<Map<String, dynamic>> results;
  final Size previewSize;
  final Size screenSize;
  final List<String> cocoLabels;

  const BoundingBoxOverlay({
    Key? key,
    required this.results,
    required this.previewSize,
    required this.screenSize,
    required this.cocoLabels,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    List<Widget> boxes = [];
    for (var result in results) {
      var rect = result['rect'] as List;
      int cls = result['class'];

      double top = rect[0] * screenSize.height;
      double left = rect[1] * screenSize.width;
      double bottom = rect[2] * screenSize.height;
      double right = rect[3] * screenSize.width;
      double width = right - left;
      double height = bottom - top;

      boxes.add(Positioned(
        left: left,
        top: top,
        width: width,
        height: height,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: Colors.red, width: 2),
          ),
          child: Center(
            child: Container(
              color: Colors.red,
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              child: Text(
                '${cls < cocoLabels.length ? cocoLabels[cls] : "class $cls"}\n${(result['score'] * 100).toStringAsFixed(0)}%',
                style: const TextStyle(color: Colors.white, fontSize: 12),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ),
      ));
    }
    return Stack(children: boxes);
  }
}