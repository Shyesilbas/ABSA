using System.Net.Http.Headers;
using System.Text;
using BackendGateway.Models;
using Microsoft.Extensions.Options;

namespace BackendGateway.Services;

public sealed class PythonInferenceClient
{
    private readonly HttpClient _httpClient;
    private readonly GatewayOptions _options;

    public PythonInferenceClient(HttpClient httpClient, IOptions<GatewayOptions> options)
    {
        _httpClient = httpClient;
        _options = options.Value;
    }

    public async Task<HttpResponseMessage> GetAsync(string path, CancellationToken ct)
    {
        var req = BuildRequest(HttpMethod.Get, path);
        return await _httpClient.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
    }

    public async Task<HttpResponseMessage> PostJsonAsync(string path, string jsonBody, CancellationToken ct)
    {
        var req = BuildRequest(HttpMethod.Post, path);
        req.Content = new StringContent(jsonBody, Encoding.UTF8, "application/json");
        return await _httpClient.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
    }

    public async Task<HttpResponseMessage> PostMultipartAsync(
        string path,
        IFormFile file,
        string? topicTitle,
        string? keywordsSubtitle,
        CancellationToken ct)
    {
        var req = BuildRequest(HttpMethod.Post, path);
        var content = new MultipartFormDataContent();

        await using var stream = file.OpenReadStream();
        var fileContent = new StreamContent(stream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue(file.ContentType ?? "text/csv");
        content.Add(fileContent, "file", file.FileName);

        if (!string.IsNullOrWhiteSpace(topicTitle))
        {
            content.Add(new StringContent(topicTitle), "topic_title");
        }

        if (!string.IsNullOrWhiteSpace(keywordsSubtitle))
        {
            content.Add(new StringContent(keywordsSubtitle), "keywords_subtitle");
        }

        req.Content = content;
        return await _httpClient.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
    }

    private HttpRequestMessage BuildRequest(HttpMethod method, string path)
    {
        var request = new HttpRequestMessage(method, path.StartsWith("/") ? path : $"/{path}");
        if (!string.IsNullOrWhiteSpace(_options.InternalInferenceToken))
        {
            request.Headers.TryAddWithoutValidation("x-inference-token", _options.InternalInferenceToken);
        }

        return request;
    }
}
