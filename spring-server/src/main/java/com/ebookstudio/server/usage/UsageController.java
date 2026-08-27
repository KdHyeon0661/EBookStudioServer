package com.ebookstudio.server.usage;

import com.ebookstudio.server.auth.JwtPrincipal;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
public class UsageController {
    private final UsageService usage;

    public UsageController(UsageService usage) {
        this.usage = usage;
    }

    @PostMapping("/usage/events")
    public UsageService.BatchResult events(@AuthenticationPrincipal JwtPrincipal principal,
                                           @RequestBody UsageBatchRequest request) {
        return usage.ingest(principal, request == null ? null : request.events());
    }

    @GetMapping("/usage/summary")
    public UsageService.UsageSummary summary(@AuthenticationPrincipal JwtPrincipal principal) {
        return usage.summary(principal);
    }

    @GetMapping("/usage/books")
    public BookUsageResponse books(@AuthenticationPrincipal JwtPrincipal principal) {
        return new BookUsageResponse(usage.books(principal));
    }

    @GetMapping("/usage/daily")
    public UsageService.DailyUsageSeries daily(
            @AuthenticationPrincipal JwtPrincipal principal,
            @RequestParam(value = "days", defaultValue = "7") int days) {
        return usage.daily(principal, days);
    }

    public record BookUsageResponse(
            @JsonProperty("books") List<UsageService.BookUsageSummary> books) { }

    public record UsageBatchRequest(
            @JsonProperty("events") List<UsageService.UsageEventInput> events) { }
}
