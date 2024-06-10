import React from 'react';

const AuthorizedEntryTable = ({ entries }) => {
  return (
    <div className="authorized-entry-table">
      <h2>Unauthorized Access Violations</h2>
      <table>
        <thead>
          <tr>
            <th>Violation</th>
            <th>Details</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry, index) => (
            <tr key={index}>
              <td>{entry.violation}</td>
              <td>{entry.details}</td>
              <td>{entry.timestamp}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AuthorizedEntryTable;
