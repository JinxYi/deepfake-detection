import { Post } from "@/interface/Post";
import React, { createContext, useState, useContext, useEffect } from "react";

interface PostContextProps {
  posts: Post[];
  addPost: (post: Post) => void;
}

const PostContext = createContext<PostContextProps | undefined>(undefined);
interface PostProviderProps {
  children: React.ReactNode;
}
export const PostProvider = ({ children }: PostProviderProps) => {
  const [posts, setPosts] = useState<Post[]>([]);

  useEffect(() => {
    getPosts();
  }, []);

  const getPosts = () => {
    const post = fetchPost();
    setPosts([post]);
  };

  const addPost = (post: Post) => {
    setPosts([...posts, post]);
  };

  return (
    <PostContext.Provider value={{ posts, addPost }}>
      {children}
    </PostContext.Provider>
  );
};

export const usePosts = () => {
  const context = useContext(PostContext);
  if (!context) {
    throw new Error("usePosts must be used within a PostProvider");
  }
  return context;
};

function fetchPost() {
  return {
    id: 1,
    owner: "jinx",
    date: new Date(Date.now()),
    content:
      "This impressive paella is a perfect party dish and a fun meal to cook together with your guests. Add 1 cup of frozen peas along with the mussels, if you like",
  };
}
