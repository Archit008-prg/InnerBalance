FROM node:18-alpine

WORKDIR /app

# Cache package dependencies install
COPY package*.json ./
RUN npm install

# Copy frontend source code
COPY . .

# Build Next.js application
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
