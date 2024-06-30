import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Graph = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/get-unauthorized-face-screenshots');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Count entries by month
        const monthlyCounts = countEntriesByMonth(data);

        // Transform data into format required by Recharts LineChart
        const chartData = Object.keys(monthlyCounts).map((month) => ({
          name: month,
          count: monthlyCounts[month],
        }));

        setData(chartData);
      } catch (error) {
        console.error('Error fetching unauthorized access data:', error);
      }
    };

    // Fetch data immediately when component mounts
    fetchData();

    // Set interval to fetch data every 30 seconds
    const interval = setInterval(fetchData, 30000);

    // Clean up interval on component unmount
    return () => clearInterval(interval);
  }, []);

  // Function to count entries by month
  const countEntriesByMonth = (entries) => {
    const counts = {};
    entries.forEach((entry) => {
      const month = new Date(entry.timestamp).toLocaleString('en-US', { month: 'short' });
      if (counts[month]) {
        counts[month] += 1;
      } else {
        counts[month] = 1;
      }
    });
    return counts;
  };

  return (
    <div className="graph">
      <h2>Unauthorized Accesses by Employee</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="count" stroke="#8884d8" activeDot={{ r: 8 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default Graph;
