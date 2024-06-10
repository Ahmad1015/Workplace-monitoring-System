import React from 'react';
import EmployeeTable from './components/EmployeeTable';
import Graph from './components/Graph';
import ScreenshotSection from './components/ScreenshotSection';
import AuthorizedEntryTable from './components/AuthorizedEntryTable';
import './App.css';

const employees = [
  { id: '2365', name: 'Wajahat Mahmood Qazi', zone: 'Zone 1', status: 'Not Allowed' },
  { id: '2110', name: 'Murtaza', zone: 'Zone 3', status: 'Allowed' },
  { id: '1325', name: 'Syed Tanweer', zone: 'Zone 2', status: 'Allowed' },
];

const violations = [
  { violation: 'Unauthorized Entry', details: 'Zone 1 Entry', timestamp: '2023-06-10 08:47 AM' },
  { violation: 'Suspicious Activity', details: 'Zone 2 Entry', timestamp: '2023-06-10 09:47 AM' },
];

const screenshots = [
  { url: 'https://via.placeholder.com/150', timestamp: '2023-06-10 08:47 AM' },
  { url: 'https://via.placeholder.com/150', timestamp: '2023-06-10 09:47 AM' },
];

const App = () => {
  return (
    <div className="App">
      <header className="App-header">
        <h2>Policy-Based Presence Tracking</h2>
      </header>
      <div className="content">
        <div className="tables">
          <EmployeeTable title="Policy-based Workplace Safety" employees={employees} />
          <AuthorizedEntryTable entries={violations} />
        </div>
        <Graph />
        <ScreenshotSection screenshots={screenshots} />
      </div>
    </div>
  );
}

export default App;
