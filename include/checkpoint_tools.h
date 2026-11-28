#ifndef CHECKPOINT_TOOLS_H
#define CHECKPOINT_TOOLS_H

#include <deal.II/base/mpi.h>

#include <string>

/**
 * Tools to manage the checkpoint directories.
 *
 * The checkpoint written at time step N is stored in its own directory
 * <output_dir><prefix>_<N>.chk/, with N padded with zeros to 6 digits. It is
 * first written to the temporary directory <prefix>_<N>.chk.tmp/, which is
 * renamed once all processes are done writing. A directory <prefix>_<N>.chk
 * thus always holds a complete checkpoint, even if the simulation was
 * interrupted while writing a checkpoint.
 *
 * The file operations are performed on the root process, and their errors
 * are thrown on all processes.
 */
namespace CheckpointTools
{
  /**
   * Return the path of the directory of the checkpoint written at time step
   * @p time_step, with a trailing slash.
   */
  std::string checkpoint_directory(const std::string &output_dir,
                                   const std::string &prefix,
                                   const unsigned int time_step);

  /**
   * Create the empty temporary directory to which the checkpoint at time step
   * @p time_step is written, and return its path, with a trailing slash. A
   * temporary directory left by an interrupted checkpoint is removed first.
   */
  std::string
  create_temporary_checkpoint_directory(const std::string &output_dir,
                                        const std::string &prefix,
                                        const unsigned int time_step,
                                        const MPI_Comm     comm);

  /**
   * Rename the temporary directory of the checkpoint at time step
   * @p time_step to the checkpoint directory, once all processes are done
   * writing to it.
   *
   * If the checkpoint directory of this time step already exists, e.g., when
   * restarting from an earlier checkpoint, it is first renamed to
   * <prefix>_<N>.bak<k>.chk, with k the first unused index.
   *
   * If @p n_checkpoints_to_keep is positive, only the
   * @p n_checkpoints_to_keep most recent checkpoint directories
   * <prefix>_<M>.chk with M <= @p time_step are then kept. The directories
   * with M > @p time_step, which come from an abandoned run, and the backups
   * are never removed.
   */
  void finalize_checkpoint_directory(const std::string &output_dir,
                                     const std::string &prefix,
                                     const unsigned int time_step,
                                     const unsigned int n_checkpoints_to_keep,
                                     const MPI_Comm     comm);

  /**
   * Return the path of the checkpoint directory to restart from, with a
   * trailing slash. If @p restart_from is "latest", this is the directory
   * <prefix>_<N>.chk of @p output_dir with the largest N. Otherwise,
   * @p restart_from is the path of the directory, relative to @p output_dir
   * unless it is absolute. Throws if the directory does not exist.
   */
  std::string find_restart_directory(const std::string &output_dir,
                                     const std::string &prefix,
                                     const std::string &restart_from,
                                     const MPI_Comm     comm);

  /**
   * Write the file info.txt in the checkpoint @p directory, with the time
   * step, the time and the number of processes of @p comm.
   */
  void write_checkpoint_info(const std::string &directory,
                             const unsigned int time_step,
                             const double       time,
                             const MPI_Comm     comm);

  /**
   * Check that the checkpoint in @p directory was written with as many
   * processes as in @p comm, since each process reads the solution vectors
   * it wrote.
   */
  void check_checkpoint_info(const std::string &directory, const MPI_Comm comm);
} // namespace CheckpointTools

#endif
