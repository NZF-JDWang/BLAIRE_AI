FROM node:22-alpine

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY . .
RUN npm run build
RUN npm prune --omit=dev

USER appuser

EXPOSE 3000

CMD ["npm", "run", "start"]
