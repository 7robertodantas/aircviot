import 'dart:async';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

import 'mqtt_config.dart';

class ServicoMQTT {
  static late final MqttServerClient _cliente;
  static bool _conectado = false;
  static int _tentativas = 0;
  static const int _maximoTentativas = 3;
  static String? _ultimoValorEnviado;

  static Future<void> conectar() async {
    _cliente = MqttServerClient(MQTT_HOST, MQTT_CLIENT_ID)
      ..port = MQTT_PORT
      ..logging(on: false)
      ..keepAlivePeriod = 20
      ..onConnected = _aoConectar
      ..onDisconnected = _aoDesconectar
      ..onSubscribed = _aoAssinar
      ..onSubscribeFail = _aoFalhaAssinar;

    _cliente.connectionMessage = MqttConnectMessage()
        .authenticateAs(MQTT_USERNAME, MQTT_PASSWORD)
        .withClientIdentifier(MQTT_CLIENT_ID)
        .startClean()
        .withWillQos(MqttQos.atLeastOnce);

    while (_tentativas < _maximoTentativas) {
      try {
        print('Tentando conectar... tentativa ${_tentativas + 1}');
        await _cliente.connect();

        if (_cliente.connectionStatus?.state == MqttConnectionState.connected) {
          _conectado = true;
          _publicarDisponibilidade("online");
          break;
        } else {
          throw Exception('Estado da conexão: ${_cliente.connectionStatus?.state}');
        }
      } catch (e) {
        print('Erro ao conectar: $e');
        _cliente.disconnect();
        _tentativas++;
        await Future.delayed(const Duration(seconds: 2));
      }
    }

    if (!_conectado) {
      print('❌ Não foi possível conectar ao servidor MQTT após $_maximoTentativas tentativas.');
    }
  }

  static void publicar(String mensagem) {
    if (!_conectado) {
      print('⚠️ MQTT não está conectado. Mensagem não enviada.');
      return;
    }

    if (mensagem == _ultimoValorEnviado) {
      print('🔁 Valor repetido. Ignorando envio.');
      return;
    }

    final builder = MqttClientPayloadBuilder();
    builder.addString(mensagem);
    _cliente.publishMessage(MQTT_TOPIC_ESTADO, MqttQos.atLeastOnce, builder.payload!);
    _ultimoValorEnviado = mensagem;
    print('📤 Mensagem publicada no tópico $MQTT_TOPIC_ESTADO: $mensagem');
  }

  static void _publicarDisponibilidade(String status) {
    final builder = MqttClientPayloadBuilder();
    builder.addString(status);
    _cliente.publishMessage("aha/ESP32_Wokwi_01/avty_t", MqttQos.atLeastOnce, builder.payload!, retain: true);

    print('📶 Disponibilidade publicada: $status');
  }

  static void setDisponibilidadeOnline() => _publicarDisponibilidade("online");
  static void setDisponibilidadeOffline() => _publicarDisponibilidade("offline");

  static void desconectar() {
    _publicarDisponibilidade("offline");
    _cliente.disconnect();
    _conectado = false;
    print('🔌 Cliente MQTT desconectado manualmente.');
  }

  static void _aoConectar() => print('✅ Conectado ao MQTT com sucesso.');
  static void _aoDesconectar() => print('🔌 Conexão com o MQTT foi encerrada.');
  static void _aoAssinar(String topico) => print('📡 Assinado ao tópico: $topico');
  static void _aoFalhaAssinar(String topico) => print('❌ Falha ao assinar tópico: $topico');
}