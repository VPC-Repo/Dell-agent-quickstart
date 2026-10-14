# The installer behind `docker compose up`: the Docker CLI and Compose plugin, the tools the Intel toolkit's
# deployment script uses, and Python for the helper scripts. It drives the host's Docker through the socket.
FROM debian:bookworm-slim
ARG YQ_VERSION=v4.44.3
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends ca-certificates curl gnupg git jq openssl python3 procps iproute2 \
      gosu sudo lsb-release hostname \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
 && apt-get update -qq && apt-get install -y -qq --no-install-recommends docker-ce-cli docker-compose-plugin docker-buildx-plugin \
 && curl -fsSL -o /usr/local/bin/yq "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_$(dpkg --print-architecture)" \
 && chmod +x /usr/local/bin/yq \
 && rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /usr/local/bin/installer-entrypoint
ENTRYPOINT ["installer-entrypoint"]
