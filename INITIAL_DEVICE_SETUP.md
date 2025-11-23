# Set up a new device 

# SSH into the device

> ssh -X <username>@192.168.55.1

Provide your password (default `qwe`)

# Set up the wifi (2.4GHz networks only)

> sudo nmcli dev wifi connect <SSID> password <your_wifi_password>

# Clone the code

Configure to store your github credentials upon first login

> git config --global credential.helper store

You will need your github username and the Personal Access Token here

> git clone https://github.com/PaulChouLedger/LedgerAI.git

# Copy in the model files

> scp -r C:/Users/thinh/Documents/TN-Code/LedgerAI/llm-container/models thinh@192.168.55.1:/home/thinh/LedgerAI/llm-container/models

# Troubleshooting

1. `Docker` - need to use `sudo` for docker

> sudo docker ps



