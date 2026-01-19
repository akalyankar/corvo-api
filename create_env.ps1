# PowerShell script to create .env file with proper UTF-8 encoding
$envContent = @"
# Corvo API Configuration
CORVO_PATH=../corvo
FLASK_HOST=0.0.0.0
FLASK_PORT=8000
FLASK_DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_COLLECTION=corvo
CORS_ORIGINS=*
"@

# Write with UTF-8 encoding (no BOM)
[System.IO.File]::WriteAllText("$PSScriptRoot\.env", $envContent, [System.Text.UTF8Encoding]::new($false))
Write-Host ".env file created successfully with UTF-8 encoding"

