import React, { useState } from 'react';

const ScreenshotSection = ({ screenshots }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const nextScreenshot = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % screenshots.length);
  };

  const prevScreenshot = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + screenshots.length) % screenshots.length);
  };

  return (
    <div className="screenshot-section">
      <div className="screenshot">
        <img src={screenshots[currentIndex].url} alt={`Screenshot ${currentIndex + 1}`} />
        <p>{screenshots[currentIndex].timestamp}</p>
      </div>
      <div className="nav-buttons">
        <button className="nav-button" onClick={prevScreenshot}>Previous</button>
        <button className="nav-button" onClick={nextScreenshot}>Next</button>
      </div>
    </div>
  );
};

export default ScreenshotSection;
