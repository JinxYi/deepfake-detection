import React, { createContext, useState, useContext, ReactNode } from "react";

interface DrawerContextProps {
  drawerWidth: number;
  setDrawerWidth: (width: number) => void;
}

const DrawerContext = createContext<DrawerContextProps | undefined>(undefined);
interface DrawerProviderProps {
  children: ReactNode;
}
export const DrawerProvider = ({ children }: DrawerProviderProps) => {
  const [drawerWidth, setDrawerWidth] = useState(240);

  return (
    <DrawerContext.Provider value={{ drawerWidth, setDrawerWidth }}>
      {children}
    </DrawerContext.Provider>
  );
};

export const useDrawer = () => {
  const context = useContext(DrawerContext);
  if (!context) {
    throw new Error("useDrawer must be used within a DrawerProvider");
  }
  return context;
};
