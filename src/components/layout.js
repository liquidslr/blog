import * as React from 'react';
import { Link } from 'gatsby';
import { ThemeToggler } from 'gatsby-plugin-dark-mode';
import darkIcon from '../images/night-mode.png';
import lightIcon from '../images/day-mode.png';

const Layout = ({ location, title, children }) => {
  const rootPath = `${__PATH_PREFIX__}/`;
  const isRootPath = location.pathname === rootPath;
  let header;

  if (isRootPath) {
    header = (
      <h1 className="main-heading">
        <Link to="/">{title}</Link>
      </h1>
    );
  } else {
    header = (
      <Link className="header-link-home" to="/">
        {title}
      </Link>
    );
  }

  return (
    <div className="global-wrapper" data-is-root-path={isRootPath}>
      <div
        className="flex"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <header className="global-header">{header}</header>
        <div>
          <ThemeToggler>
            {({ theme, toggleTheme }) =>
              theme === 'dark' ? (
                <img
                  onClick={(e) => toggleTheme('light')}
                  src={lightIcon}
                  width={'25'}
                />
              ) : (
                <img
                  onClick={(e) => toggleTheme('dark')}
                  src={darkIcon}
                  width={'25'}
                />
              )
            }
          </ThemeToggler>
        </div>
      </div>
      <main>{children}</main>
    </div>
  );
};

export default Layout;
