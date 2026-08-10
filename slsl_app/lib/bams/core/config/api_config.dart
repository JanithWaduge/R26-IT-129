class ApiConfig {
  const ApiConfig._();

  static const String baseUrl = String.fromEnvironment(
     'API_BASE_URL',
     defaultValue: 'http://10.0.2.2:8000/api/v1',
   );
  static String resolveMediaUrl(String mediaUri) {
    final Uri parsedMedia = Uri.parse(mediaUri);

    if (parsedMedia.hasScheme) {
      return mediaUri;
    }

    final Uri apiUri = Uri.parse(baseUrl);

    final Uri origin = Uri(
      scheme: apiUri.scheme,
      host: apiUri.host,
      port: apiUri.hasPort ? apiUri.port : null,
    );

    return origin.resolve(mediaUri).toString();
  }
}
