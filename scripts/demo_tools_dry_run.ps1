Write-Host "Wave 1 dry-run demo commands"
Write-Host 'python -m blaire_core.main --env dev "/tool host_health_snapshot {\"host\":\"bsl1\"}"'
Write-Host 'python -m blaire_core.main --env dev "/tool obsidian_search {\"query\":\"homelab\"}"'
Write-Host 'python -m blaire_core.main --env dev "/tool media_pipeline_status {}"'
Write-Host 'python -m blaire_core.main --env dev "/tool docker_container_restart {\"host\":\"bsl1\",\"container\":\"jellyfin\"}"'
Write-Host 'python -m blaire_core.main --env dev "/approvals list"'
Write-Host 'python -m blaire_core.main --env dev "/approve <token> docker_container_restart {\"host\":\"bsl1\",\"container\":\"jellyfin\"}"'
