import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

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
      title: 'TFLite Flutter Detection Debug',
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
  int inputSize = 300; // SSD MobileNet default
  bool _isDetecting = false;
  List<Map<String, dynamic>> _results = [];
  Timer? _timer;

  final List<String> cocoLabels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light"
    // ... add more if you wish!
  ];

  @override
  void initState() {
    super.initState();
    print("App started, loading model...");
    _initEverything();
  }

  Future<void> _initEverything() async {
    await _loadModel();
    await _initCamera(_selectedCameraIdx);
    print("Model and camera fully initialized! Starting detection timer.");
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) => _runDetection());
  }

  Future<void> _loadModel() async {
    print("Loading TFLite model...");
    _interpreter = await Interpreter.fromAsset('assets/ssd_mobilenet.tflite');
    print("TFLite model loaded.");
    setState(() {});
  }

  Future<void> _initCamera(int idx) async {
    print("Initializing camera index $idx...");
    _cameraController?.dispose();
    _cameraController = CameraController(
      cameras[idx],
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await _cameraController!.initialize();
    print("Camera initialized!");
    setState(() {});
  }

  void _switchCamera() {
    int newIdx = (_selectedCameraIdx + 1) % cameras.length;
    print("Switching to camera index $newIdx...");
    setState(() {
      _selectedCameraIdx = newIdx;
      _initCamera(newIdx);
    });
  }

  Future<void> _runDetection() async {
    print("Running detection...");
    if (_interpreter == null || _cameraController == null || _isDetecting) {
      print("Detection skipped: Interpreter or camera not ready.");
      return;
    }
    if (!_cameraController!.value.isInitialized) {
      print("Camera not initialized.");
      return;
    }

    _isDetecting = true;
    try {
      XFile file = await _cameraController!.takePicture();
      print("Image captured from camera.");
      final bytes = await file.readAsBytes();

      final img.Image? image = img.decodeImage(bytes);
      if (image == null) {
        print("Failed to decode image!");
        _isDetecting = false;
        return;
      }
      // Resize to model input (300x300)
      final img.Image resized = img.copyResize(image, width: inputSize, height: inputSize);
      print("Image resized to $inputSize x $inputSize for model.");

      // Uint8 input
      var input = imageToByteListUint8(resized, inputSize);
      var inputList = input.reshape([1, inputSize, inputSize, 3]);

      // Model outputs
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
      print("Model executed. Detection count: $count");

      List<Map<String, dynamic>> detections = [];
      for (int i = 0; i < count; i++) {
        int cls = (classes[0][i] as double).toInt();
        double score = scores[0][i];
        var rect = boxes[0][i];
        print('Detection $i: class: $cls (${cls < cocoLabels.length ? cocoLabels[cls] : "?"}), score: ${score.toStringAsFixed(2)}, rect: $rect');
        if (cls == 0 && score > 0.5) { // Only 'person'
          detections.add({
            'score': score,
            'rect': rect,
            'class': cls,
          });
        }
      }
      setState(() {
        _results = detections;
      });
    } catch (e) {
      print('Detection error: $e');
    }
    _isDetecting = false;
  }

  @override
  void dispose() {
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

/// This overlay version corrects bounding box aspect ratio for camera preview/screen
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
    // Calculate scale/offset to match camera preview
    final double previewAspectRatio = previewSize.width / previewSize.height;
    final double screenAspectRatio = screenSize.width / screenSize.height;

    double scaleX, scaleY, dx = 0, dy = 0;
    if (screenAspectRatio > previewAspectRatio) {
      // Screen is wider than preview: fit height, pad width
      scaleY = screenSize.height;
      scaleX = previewAspectRatio * screenSize.height;
      dx = (screenSize.width - scaleX) / 2;
    } else {
      // Screen is taller than preview: fit width, pad height
      scaleX = screenSize.width;
      scaleY = screenSize.width / previewAspectRatio;
      dy = (screenSize.height - scaleY) / 2;
    }

    List<Widget> boxes = [];
    for (var result in results) {
      var rect = result['rect'] as List;
      int cls = result['class'];

      // Model returns [top, left, bottom, right] normalized (0-1)
      double top = rect[0] * scaleY + dy;
      double left = rect[1] * scaleX + dx;
      double bottom = rect[2] * scaleY + dy;
      double right = rect[3] * scaleX + dx;
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
          child: Align(
            alignment: Alignment.topLeft,
            child: Container(
              color: Colors.red,
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              child: Text(
                '${cls < cocoLabels.length ? cocoLabels[cls] : "class $cls"}\n${(result['score'] * 100).toStringAsFixed(0)}%',
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ),
          ),
        ),
      ));
    }
    return Stack(children: boxes);
  }
}
