import React, { useState, useEffect, useRef } from 'react';
import './ScreenshotGallery.css'; // Import your CSS file

const ScreenshotGallery = () => {
  const [mediaItems, setMediaItems] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    const fetchScreenshots = async () => {
      try {
        const response = await fetch('http://localhost:8000/get-face-screenshots');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Screenshots data:', data); // Debugging line
        return data;
      } catch (error) {
        console.error('Error fetching screenshot data:', error);
        return [];
      }
    };

    const fetchVideos = async () => {
      try {
        const response = await fetch('http://localhost:8000/get-videos');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Videos data:', data); // Debugging line
        return data;
      } catch (error) {
        console.error('Error fetching video data:', error);
        return [];
      }
    };

    const fetchData = async () => {
      const screenshots = await fetchScreenshots();
      const videos = await fetchVideos();
      const combinedMedia = [...screenshots, ...videos].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
      setMediaItems(combinedMedia);
    };

    fetchData();

    intervalRef.current = setInterval(fetchData, 5000);

    return () => {
      clearInterval(intervalRef.current);
    };
  }, []);

  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % mediaItems.length);
  };

  const handlePrev = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + mediaItems.length) % mediaItems.length);
  };

  const renderMedia = (item) => {
    if (item.screenshot_path) {
      const imgPath = `http://localhost:8000${item.screenshot_path}`;
      console.log('Image path:', imgPath); // Debugging line
      return (
        <img
          src={imgPath}
          alt={`Screenshot ${currentIndex}`}
          className="screenshot-img"
          onError={(e) => {
            console.error('Error loading image:', e.target.src); // Debugging line
          }}
        />
      );
    } else if (item.video_path) {
      const videoPath = `http://localhost:8000${item.video_path}`;
      console.log('Video path:', videoPath); // Debugging line
      return (
        <video
          src={videoPath}
          alt={`Video ${currentIndex}`}
          className="screenshot-img"
          controls
          onError={(e) => {
            console.error('Error loading video:', e.target.src); // Debugging line
          }}
        >
          Your browser does not support the video tag.
        </video>
      );
    }
    return null;
  };

  return (
    <div className="screenshot-section">
      <div className="screenshots">
        <div className="screenshot">
          {mediaItems.length > 0 && renderMedia(mediaItems[currentIndex])}
        </div>
      </div>
      <div className="navigation-buttons">
        <button onClick={handlePrev}>Previous</button>
        <button onClick={handleNext}>Next</button>
      </div>
    </div>
  );
};

export default ScreenshotGallery;
