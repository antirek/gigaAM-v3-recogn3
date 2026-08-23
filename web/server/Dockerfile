FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json .npmrc ./
RUN npm ci --omit=dev

COPY src ./src
COPY scripts ./scripts

ENV WEB_PORT=3000
EXPOSE 3000

CMD ["node", "src/index.js"]
