# =====================================================================
# Qwen3.8-27B 双方案一键配置与测试脚本 (PowerShell)
# =====================================================================

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Qwen3.8-27B 本地部署管理器 (AMD RX 9070 XT 专属优化)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "请选择要执行的操作："
Write-Host "[1] 构建并启动方案一：Q3_K_M (100% 显存全速模式，~12.2GB)" -ForegroundColor Green
Write-Host "[2] 构建并启动方案二：Q4_K_M (80% 显存 + 20% 内存混合模式，~15.5GB)" -ForegroundColor Yellow
Write-Host "[3] 测试已部署的 27B 模型推理速度与连通性" -ForegroundColor Magenta
Write-Host "[0] 退出"
Write-Host ""

$choice = Read-Host "请输入选项 (1/2/3/0)"

$deployDir = Join-Path $PSScriptRoot "..\deploy"

switch ($choice) {
    "1" {
        Write-Host "`n🚀 正在构建方案一: qwen3.8:27b-q3 (纯显存全速版)..." -ForegroundColor Green
        $modelfile = Join-Path $deployDir "qwen3.8_27b_q3_km.Modelfile"
        ollama create qwen3.8:27b-q3 -f $modelfile
        Write-Host "✅ 构建完毕！立即启动交互体验..." -ForegroundColor Green
        ollama run qwen3.8:27b-q3
    }
    "2" {
        Write-Host "`n🚀 正在构建方案二: qwen3.8:27b-q4 (显存+内存混合版)..." -ForegroundColor Yellow
        $modelfile = Join-Path $deployDir "qwen3.8_27b_q4_km.Modelfile"
        ollama create qwen3.8:27b-q4 -f $modelfile
        Write-Host "✅ 构建完毕！立即启动交互体验..." -ForegroundColor Yellow
        ollama run qwen3.8:27b-q4
    }
    "3" {
        Write-Host "`n🔍 正在向本地 Ollama 发送基准测试问答..." -ForegroundColor Magenta
        $testPrompt = "你好！请用两句话介绍一下你自己，并写一段简单的 Python 快速排序代码。"
        $body = @{
            model = "qwen3.8:27b-q3"
            messages = @(@{ role = "user"; content = $testPrompt })
            stream = $false
        } | ConvertTo-Json

        try {
            $t0 = Get-Date
            $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/chat" -Method Post -Body $body -ContentType "application/json"
            $t1 = Get-Date
            $cost = ($t1 - $t0).TotalSeconds
            Write-Host "耗时: $($cost.ToString('F2')) 秒" -ForegroundColor Green
            Write-Host "模型回复内容:`n" -ForegroundColor White
            Write-Host $resp.message.content -ForegroundColor Cyan
        } catch {
            Write-Host "⚠️ 请求失败，请检查 Ollama 服务是否启动 (http://localhost:11434)" -ForegroundColor Red
        }
    }
    Default {
        Write-Host "已退出。"
    }
}

