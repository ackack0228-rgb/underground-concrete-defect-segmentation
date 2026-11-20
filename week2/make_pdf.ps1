# week2_submission.docx 를 "있는 그대로" 제출용 PDF로 변환한다.
# docx 내용은 전혀 건드리지 않으므로, Word에서 직접 고친 뒤 몇 번이든 다시 실행해도 안전하다.
# 실행: make_pdf.bat 더블클릭 (또는 PowerShell에서 .\make_pdf.ps1)

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
[string]$docx = Join-Path $dir 'week2_submission.docx'
[string]$pdf  = Join-Path $dir '21113103_컴퓨터공학전공_김근영.pdf'

if (-not (Test-Path $docx)) { Write-Host "docx 파일이 없습니다: $docx"; exit 1 }

# Word에서 열어둔 채로 실행하면 "마지막으로 저장한 내용"까지만 PDF에 들어간다.
if (Test-Path (Join-Path $dir '~$ek2_submission.docx')) {
    Write-Host '주의: 문서가 Word에 열려 있습니다. 저장(Ctrl+S)한 내용까지만 PDF에 반영됩니다.'
}

$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($docx, $false, $true)   # 읽기 전용으로 열기
    $doc.ExportAsFixedFormat($pdf, 17)                  # 17 = wdExportFormatPDF
    $pages = $doc.ComputeStatistics(2)                  # 2 = wdStatisticPages
    $doc.Close($false)
    Write-Host "PDF 저장 완료 ($pages 쪽): $pdf"
} catch {
    Write-Host "변환 실패: $($_.Exception.Message)"
    Write-Host 'PDF 파일이 다른 프로그램(뷰어·브라우저)에 열려 있으면 닫고 다시 실행하세요.'
    exit 1
} finally {
    $word.Quit()
}
