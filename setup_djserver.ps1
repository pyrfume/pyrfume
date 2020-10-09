Function Test-CommandExists
{
    Param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try { 
        if (Get-Command $command) { 
            RETURN $true
        } 
    } Catch {
        RETURN $false
    } Finally { 
        $ErrorActionPreference = $oldPreference 
    }
}

If (-Not (Test-Path -Path mysql-docker) ) {
    New-Item -ItemType Directory -Name  mysql-docker
}

If (-Not (Test-CommandExists docker-compose)) {
    Write-Host "docker-compose does not exist.";
    Write-Host "Please install docker and docker-compose before running this script.";
    exit 1;
}

If (-Not (Test-Path -Path mysql-docker/docker-compose.yml) ) {
    $url = "https://raw.githubusercontent.com/datajoint/mysql-docker/master/slim/docker-compose.yml";
    $output = "mysql-docker/docker-compose.yml";
    $client = New-Object System.Net.WebClient;
    $client.DownloadFile($url, $output);
}

docker-compose up -d
