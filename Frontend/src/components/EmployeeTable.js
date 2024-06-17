import React from 'react';

const EmployeeTable = ({ title, employees }) => {
  return (
    <div className="employee-table">
      <h2>{title}</h2>
      <table>
        <thead>
          <tr>
            <th>Employee</th>
            <th>Full Name</th>
            <th>Zone</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {employees.map((employee) => (
            <tr key={employee.id}>
              <td>{employee.id}</td>
              <td>{employee.name}</td>
              <td>{employee.zone}</td>
              <td>{employee.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default EmployeeTable;
