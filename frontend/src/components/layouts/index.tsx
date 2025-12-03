export * from "./appbar";
export * from "./sidebar";

import { DrawerProvider } from "@/context/drawer-context";
import { Sidebar } from "./sidebar";
export const Layout = ({ children }: React.PropsWithChildren) => {
  return (
    <DrawerProvider>
      <Sidebar title="Deepfake Detector">{children}</Sidebar>
    </DrawerProvider>
  );
};
