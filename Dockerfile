# Use the official lightweight Python image.
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Create a non-root user (good security practice and helps with some Cloud Run environments)
RUN useradd -m -r appuser && chown -R appuser /app

# Copy the requirements file first for Docker caching
COPY requirements.txt .

# Install dependencies before copying the rest of the code (saves time on rebuilds)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all local files
COPY . .

# Assume ownership of the files
RUN chown -R appuser:appuser /app

USER appuser

# Expose port (Cloud Run sets the PORT env var directly, defaulting to 8080)
EXPOSE 8080

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
