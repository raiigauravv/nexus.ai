FROM node:20-alpine

WORKDIR /app

# Install dependencies based on the preferred package manager
COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY . .

# Next.js telemetry disable
ENV NEXT_TELEMETRY_DISABLED=1

# Build Next.js
RUN npm run build

# Expose the frontend port
EXPOSE 3000

CMD ["npm", "start"]
