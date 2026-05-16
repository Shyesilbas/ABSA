using BackendGateway.Models;
using BackendGateway.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using System.Text;
using System.Text.Json;

namespace BackendGateway.Controllers;

[ApiController]
[Route("")]
public sealed class GatewayController : ControllerBase
{
    private readonly PythonInferenceClient _inferenceClient;
    private readonly GatewayOptions _options;

    public GatewayController(PythonInferenceClient inferenceClient, IOptions<GatewayOptions> options)
    {
        _inferenceClient = inferenceClient;
        _options = options.Value;
    }

    [HttpGet("health")]
    public async Task<IActionResult> Health(CancellationToken ct)
    {
        var upstream = await _inferenceClient.GetAsync("/health", ct);
        return await ToActionResult(upstream, ct);
    }

    [HttpGet("meta")]
    public async Task<IActionResult> Meta(CancellationToken ct)
    {
        var upstream = await _inferenceClient.GetAsync("/meta", ct);
        return await ToActionResult(upstream, ct);
    }

    [HttpPost("predict")]
    public async Task<IActionResult> Predict([FromBody] JsonElement body, CancellationToken ct)
    {
        var authError = RequireClientApiKey();
        if (authError is not null) return authError;

        var upstream = await _inferenceClient.PostJsonAsync("/predict", body.GetRawText(), ct);
        return await ToActionResult(upstream, ct);
    }

    [HttpPost("predict/batch")]
    public async Task<IActionResult> PredictBatch([FromBody] JsonElement body, CancellationToken ct)
    {
        var authError = RequireClientApiKey();
        if (authError is not null) return authError;

        var upstream = await _inferenceClient.PostJsonAsync("/predict/batch", body.GetRawText(), ct);
        return await ToActionResult(upstream, ct);
    }

    [HttpPost("predict/batch/upload")]
    [Consumes("multipart/form-data")]
    [RequestFormLimits(MultipartBodyLengthLimit = 1048576)]
    public async Task<IActionResult> PredictBatchUpload([FromForm] FileUploadRequest request, CancellationToken ct)
    {
        var authError = RequireClientApiKey();
        if (authError is not null) return authError;

        if (request.File == null || request.File.Length == 0)
            return BadRequest(new { detail = "Dosya yüklenemedi." });

        var upstream = await _inferenceClient.PostMultipartAsync(
            "/predict/batch/upload",
            request.File,
            request.TopicTitle,
            request.KeywordsSubtitle,
            ct);
        return await ToActionResult(upstream, ct);
    }

    [HttpPost("visualize/distribution")]
    public async Task<IActionResult> VisualizeDistribution([FromBody] JsonElement body, CancellationToken ct)
    {
        var authError = RequireClientApiKey();
        if (authError is not null) return authError;

        var upstream = await _inferenceClient.PostJsonAsync("/visualize/distribution", body.GetRawText(), ct);
        return await ToActionResult(upstream, ct);
    }

    [HttpPost("visualize/distribution/stats")]
    public async Task<IActionResult> VisualizeDistributionStats([FromBody] JsonElement body, CancellationToken ct)
    {
        var authError = RequireClientApiKey();
        if (authError is not null) return authError;

        var upstream = await _inferenceClient.PostJsonAsync("/visualize/distribution/stats", body.GetRawText(), ct);
        return await ToActionResult(upstream, ct);
    }

    private IActionResult? RequireClientApiKey()
    {
        if (string.IsNullOrWhiteSpace(_options.ClientApiKey))
        {
            return null;
        }

        var reqKey = Request.Headers["x-api-key"].ToString();
        if (reqKey == _options.ClientApiKey)
        {
            return null;
        }

        return Unauthorized(new { detail = "Unauthorized gateway request." });
    }

    private static async Task<IActionResult> ToActionResult(HttpResponseMessage upstream, CancellationToken ct)
    {
        var contentType = upstream.Content.Headers.ContentType?.ToString();
        var body = await upstream.Content.ReadAsByteArrayAsync(ct);
        var statusCode = (int)upstream.StatusCode;

        if (statusCode >= 400)
        {
            return new ContentResult
            {
                StatusCode = statusCode,
                ContentType = contentType ?? "application/json",
                Content = Encoding.UTF8.GetString(body),
            };
        }

        return new FileContentResult(body, contentType ?? "application/json");
    }
}
