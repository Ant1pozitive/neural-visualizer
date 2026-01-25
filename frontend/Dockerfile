# frontend/Dockerfile (multi-stage: build -> nginx static)
# Stage 1: build with Node
FROM node:18-alpine AS builder

WORKDIR /app
ENV PATH /app/node_modules/.bin:$PATH

# Copy package files and install dependencies
COPY frontend/package.json frontend/package-lock.json* ./ 
RUN npm ci --no-audit --progress=false

# Copy source
COPY frontend/ ./

# Build production assets
RUN npm run build

# Stage 2: serve with nginx
FROM nginx:stable-alpine

# Remove default nginx content
RUN rm -rf /usr/share/nginx/html/*

# Copy built files from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config to serve SPA
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
