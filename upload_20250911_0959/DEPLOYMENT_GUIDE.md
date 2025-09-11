# 🚀 EBM Project Deployment Guide

This guide will help you deploy the Energy Balance Model (EBM) project to both GitHub and Hugging Face Spaces.

## 📋 Prerequisites

- Git installed on your system
- GitHub account
- Hugging Face account
- Python 3.7+ (for local testing)

## 🔧 Step 1: GitHub Repository Setup

### 1.1 Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the repository details:
   - **Repository name**: `ebm-climate-simulator` (or your preferred name)
   - **Description**: `Interactive Energy Balance Model for climate simulation`
   - **Visibility**: Public (required for Hugging Face integration)
   - **Initialize**: Don't initialize with README (we already have one)

### 1.2 Upload to GitHub

```bash
# Navigate to your project directory
cd C:\Users\user\Downloads\ebm_project

# Add all files to Git
git add .

# Commit the changes
git commit -m "Initial commit: EBM climate simulator with heatmap visualization"

# Add your GitHub repository as remote origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

### 1.3 Repository Structure

Your GitHub repository should contain:
```
ebm-climate-simulator/
├── app.py                 # Original Gradio app
├── app_hf.py             # Hugging Face optimized version
├── colab_app.py          # Google Colab version
├── colab_setup.ipynb     # Colab setup notebook
├── ebm.py                # Core EBM model
├── test_heatmap.py       # Heatmap testing script
├── requirements.txt      # Python dependencies
├── requirements_hf.txt   # Hugging Face dependencies
├── README.md             # Main documentation
├── README_HF.md          # Hugging Face specific README
├── DEPLOYMENT_GUIDE.md   # This file
├── .gitignore            # Git ignore rules
└── LICENSE               # MIT License (optional)
```

## 🌐 Step 2: Hugging Face Spaces Deployment

### 2.1 Create Hugging Face Account

1. Go to [Hugging Face](https://huggingface.co)
2. Sign up for a free account
3. Verify your email address

### 2.2 Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in the details:
   - **Space name**: `ebm-climate-simulator`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free tier)
   - **Visibility**: Public

### 2.3 Upload Files to Hugging Face

#### Option A: Direct Upload (Recommended)

1. In your new Space, click "Files and versions" tab
2. Upload the following files:
   - `app_hf.py` (rename to `app.py`)
   - `requirements_hf.txt` (rename to `requirements.txt`)
   - `README_HF.md` (rename to `README.md`)

#### Option B: Git Integration

1. In your Space settings, go to "Repository" tab
2. Connect your GitHub repository
3. Set the following:
   - **App file**: `app_hf.py`
   - **Requirements file**: `requirements_hf.txt`

### 2.4 Configure Space Settings

1. Go to "Settings" tab in your Space
2. Configure:
   - **Title**: Energy Balance Model (EBM) - Climate Simulator
   - **Emoji**: 🌍
   - **Color from**: blue
   - **Color to**: green
   - **SDK**: gradio
   - **SDK version**: 4.0.0
   - **App file**: app.py
   - **Hardware**: CPU Basic

## 🔄 Step 3: Continuous Updates

### 3.1 Update GitHub Repository

```bash
# Make your changes to the code
# Then commit and push
git add .
git commit -m "Update: Add new features"
git push origin main
```

### 3.2 Update Hugging Face Space

If using Git integration, your Space will automatically update. Otherwise, manually upload new files.

## 🧪 Step 4: Testing

### 4.1 Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app_hf.py
```

### 4.2 Hugging Face Testing

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Wait for the app to build (usually 2-5 minutes)
3. Test all functionality:
   - Parameter adjustment
   - Different visualization types
   - Error handling

## 🎯 Step 5: Sharing and Promotion

### 5.1 Share Your Space

- **Public URL**: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
- **Embed Code**: Use the provided embed code for websites
- **Social Media**: Share on Twitter, LinkedIn, etc.

### 5.2 Documentation

- Update README files with usage examples
- Add screenshots of the interface
- Include scientific background information

## 🔧 Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check requirements.txt format
   - Ensure all imports are available
   - Verify app.py structure

2. **Import Errors**:
   - Make sure all dependencies are in requirements.txt
   - Check Python version compatibility

3. **Display Issues**:
   - Verify matplotlib backend settings
   - Check font configurations for Chinese characters

4. **Performance Issues**:
   - Reduce iteration count for faster computation
   - Optimize visualization code
   - Use smaller datasets for testing

### Getting Help

- **Hugging Face Community**: [Hugging Face Forums](https://discuss.huggingface.co/)
- **GitHub Issues**: Create issues in your repository
- **Documentation**: Check Gradio and Hugging Face documentation

## 📊 Monitoring and Analytics

### Hugging Face Analytics

1. Go to your Space settings
2. Check "Analytics" tab for:
   - View counts
   - User interactions
   - Performance metrics

### GitHub Analytics

1. Go to your repository
2. Check "Insights" tab for:
   - Traffic statistics
   - Clone counts
   - Star history

## 🎉 Success Checklist

- [ ] GitHub repository created and populated
- [ ] All files uploaded correctly
- [ ] Hugging Face Space created
- [ ] App builds successfully
- [ ] All visualizations work
- [ ] Parameters adjust correctly
- [ ] Error handling works
- [ ] Documentation is complete
- [ ] Public URL is accessible
- [ ] Analytics are tracking

## 🔗 Useful Links

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [GitHub Documentation](https://docs.github.com/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)

---

**Happy Deploying! 🚀🌍**
