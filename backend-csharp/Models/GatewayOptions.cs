namespace BackendGateway.Models;

public sealed class GatewayOptions
{
    public static string SectionName => "Gateway";

    public string PythonBaseUrl { get; set; } = string.Empty;

    public string InternalInferenceToken { get; set; } = string.Empty;

    public string ClientApiKey { get; set; } = string.Empty;
}
