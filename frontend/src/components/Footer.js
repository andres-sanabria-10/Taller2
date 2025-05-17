import React from 'react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer-main">
      <div className="container-fluid">
        <div className="row">
          <div className="col-md-6">
            <p className="mb-0">© {currentYear} Taller de Grafos</p>
          </div>
          <div className="col-md-6 text-end">
            <p className="mb-0">Desarrollado para Inteligencia Artificial</p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;