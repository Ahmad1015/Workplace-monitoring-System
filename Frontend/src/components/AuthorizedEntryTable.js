import React, { useState, useEffect } from 'react';

const UnauthorizedAccessTable = () => {
  const [entries, setEntries] = useState([]);

  useEffect(() => {
    const fetchEntries = async () => {
      try {
        const response = await fetch('http://localhost:8000/get-unauthorized-face-screenshots');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setEntries(data);
      } catch (error) {
        console.error('Error fetching unauthorized access data:', error);
      }
    };

    // Fetch data initially when component mounts
    fetchEntries();

    // Set up interval to fetch data every 10 seconds (adjust as needed)
    const interval = setInterval(fetchEntries, 10000); // 10 seconds interval

    // Clean up interval on component unmount
    return () => clearInterval(interval);
  }, []); // Empty dependency array ensures effect runs only on mount and unmount

  return (
    <div className="unauthorized-access-table">
      <h2>Unauthorized Access Violations</h2>
      <table>
        <thead>
          <tr>
            <th>Violator</th>
            <th>Details</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry, index) => (
            <tr key={index}>
              <td>{entry.name}</td>
              <td>{"IP Camera 1"}</td>    
              <td>{new Date(entry.timestamp).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default UnauthorizedAccessTable;
